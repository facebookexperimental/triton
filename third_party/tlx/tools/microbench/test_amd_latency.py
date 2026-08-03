import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pytest


_AMD_LATENCY_PATH = Path(__file__).with_name("amd_latency.py")
_SPEC = importlib.util.spec_from_file_location("amd_latency", _AMD_LATENCY_PATH)
amd_latency = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = amd_latency
_SPEC.loader.exec_module(amd_latency)


def test_select_cases_by_family():
    cases = amd_latency._select_cases("baseline,valu")
    names = {case.name for case in cases}
    assert "baseline.loop" in names
    assert "valu.dependent_fma_expr" in names
    assert "valu.independent_fma_expr_x4" in names
    assert all(case.family in {"baseline", "valu"} for case in cases)


def test_unknown_bench_is_rejected():
    with pytest.raises(ValueError, match="unknown --bench"):
        amd_latency._select_cases("bogus")


def test_stats_json_shape():
    stats = amd_latency._summarize([3.0, 1.0, 2.0, 4.0, 5.0])
    encoded = json.dumps({"stats": stats})
    decoded = json.loads(encoded)
    assert decoded["stats"]["median"] == 3.0
    assert decoded["stats"]["p20"] == pytest.approx(1.8)
    assert decoded["stats"]["p80"] == pytest.approx(4.2)
    assert decoded["stats"]["cv"] > 0.0
    assert decoded["stats"]["min"] == 1.0
    assert decoded["stats"]["max"] == 5.0
    assert decoded["stats"]["samples"] == [3.0, 1.0, 2.0, 4.0, 5.0]


def test_corrected_case_metadata():
    cases = {case.name: case for case in amd_latency._cases()}
    assert cases["valu.dependent_fma_expr"].ops_per_iter == 1
    assert cases["valu.dependent_fma_expr"].unit == "ticks_per_high_level_fma_expression"
    assert cases["valu.independent_fma_expr_x4"].ops_per_iter == 4
    assert cases["valu.independent_fma_expr_x4"].meta["streams"] == 4
    assert cases["lds.dependent_gather_chase_i32"].unit == "ticks_per_dependent_local_gather"
    assert cases["lds.dependent_gather_chase_i32"].meta["timed_stores"] is False
    assert cases["lds.dependent_gather_chase_i32"].expected_check(8) == amd_latency._pointer_chase_expected([0], 8)
    assert cases["lds.independent_gather_i32_x4"].ops_per_iter == 4
    assert cases["lds.independent_gather_i32_x4"].meta["timed_stores"] is False
    assert cases["global.direct_to_lds_composite_32x32"].meta["composite_per_iter"]
    assert cases["mfma.dependent_acc_32x32x32"].unit == "ticks_per_tl_dot"
    assert cases["mfma.dependent_acc_32x32x32"].launch_meta["matrix_instr_nonkdim"] == 16
    assert cases["mfma.independent_acc_32x32x32_x4"].ops_per_iter == 4
    assert all("cycles" not in case.unit for case in cases.values())


def test_s_memtime_cycle_normalization():
    assert amd_latency.SMEMTIME_HZ == pytest.approx(2.183e9)
    assert amd_latency.GFX950_MAX_SCLK_HZ == pytest.approx(2.2e9)
    assert amd_latency.CYCLES_PER_TICK == pytest.approx(1.0078, rel=1e-3)


def test_niter_is_runtime_loop_not_constexpr_unrolled():
    source = inspect.getsource(amd_latency)
    assert "NITER: tl.constexpr" not in source
    assert "range(NITER)" not in source
    assert "tl.range(0, NITER, loop_unroll_factor=1, num_stages=0)" in source


def test_lds_pointer_chase_expected_matches_cpu_sequence():
    table = amd_latency._lds_table_values()
    idx = 0
    for _ in range(13):
        idx = table[idx]
    assert amd_latency._pointer_chase_expected([0], 13) == float(idx)

    starts = [0, 7, 19, 31]
    expected_sum = 0
    for start in starts:
        idx = start
        for _ in range(13):
            idx = table[idx]
        expected_sum += idx
    assert amd_latency._pointer_chase_expected(starts, 13) == float(expected_sum)


def test_lds_source_uses_gather_and_no_timed_stores():
    source = _AMD_LATENCY_PATH.read_text()
    function_source = source.split("def _k_lds_dependent_gather_chase", 1)[1].split("@triton.jit", 1)[0]
    timed_region = function_source.split("t0 = tlx.clock64()", 1)[1].split("t1 = tlx.clock64()", 1)[0]
    assert "tlx.local_gather" in timed_region
    assert "tlx.local_store" not in timed_region


def test_lds_isa_sanity_detects_pure_timed_gather_region():
    case = {case.name: case for case in amd_latency._cases()}["lds.dependent_gather_chase_i32"]
    asm = "s_memtime\nds_read_b32 v0, v1 offset:0\ns_memtime\nds_write_b32 v0, v1 offset:0"
    sanity = amd_latency._isa_case_sanity(case, asm, "ttg.local_gather")
    assert sanity["checks"]["ttgir_local_gather"]
    assert sanity["checks"]["timed_ds_read_b32"]
    assert sanity["checks"]["no_timed_ds_write"]
    assert sanity["timed_region_instruction_counts"]["ds_read_b32"] == 1
    assert sanity["timed_region_instruction_counts"]["ds_write_b32"] == 0


def test_unsupported_result_schema():
    case = amd_latency._select_cases("global")[1]
    result = amd_latency._unsupported_result(case, RuntimeError("builtin.unrealized_conversion_cast"), 3.0)
    assert not result["ok"]
    assert result["unsupported"]
    assert "builtin.unrealized_conversion_cast" in result["error"]
    assert result["stats"] is None
    assert result["baseline_raw_ticks_per_iter"] == 3.0
    assert result["net_ticks_per_op"] is None
    assert result["diagnostic_baseline_delta_ticks_per_op"] is None
    assert result["expected_check"] is None
    assert not result["correctness_ok"]


def test_only_known_direct_to_lds_error_is_unsupported():
    direct = amd_latency._select_cases("global")[1]
    other = amd_latency._select_cases("global")[0]
    assert amd_latency._is_known_direct_to_lds_unsupported(
        direct, RuntimeError("failed to translate module to LLVM IR")
    )
    assert amd_latency._is_known_direct_to_lds_unsupported(direct, RuntimeError("builtin.unrealized_conversion_cast"))
    assert not amd_latency._is_known_direct_to_lds_unsupported(
        other, RuntimeError("failed to translate module to LLVM IR")
    )
    assert not amd_latency._is_known_direct_to_lds_unsupported(direct, RuntimeError("different compiler failure"))


def _run_or_skip_stale_clock64(**kwargs):
    try:
        return amd_latency.run_all(**kwargs)
    except RuntimeError as exc:
        if "failed to legalize operation 'ttg.clock64'" in str(exc):
            pytest.skip("active AMD extension does not include clock64 lowering; relink required")
        raise


@pytest.mark.skipif(not amd_latency.is_gfx950(), reason="requires active gfx950 HIP device")
def test_gfx950_tiny_baseline_smoke():
    result = _run_or_skip_stale_clock64(bench="baseline", niter=4, warmup=1, reps=1)
    assert result["device"]["hip"] is not None
    assert "gfx950" in result["device"]["gcn_arch_name"]
    assert result["baseline"]["name"] == "baseline.loop"
    assert len(result["benchmarks"]) == 1
    bench = result["benchmarks"][0]
    assert bench["name"] == "baseline.loop"
    assert bench["ok"]
    assert bench["stats"]["median"] > 0
    assert bench["raw_ticks_per_iter"]["median"] > 0
    assert bench["net_ticks_per_op"] is None


@pytest.mark.skipif(not amd_latency.is_gfx950(), reason="requires active gfx950 HIP device")
def test_gfx950_measured_op_latencies():
    result = _run_or_skip_stale_clock64(bench="valu,lds,global,mfma", niter=512, warmup=3, reps=7)
    benches = {bench["name"]: bench for bench in result["benchmarks"]}

    assert result["timer"]["cycles_per_tick"] == pytest.approx(1.0078, rel=1e-3)

    valu = benches["valu.dependent_fma_expr"]
    assert valu["ok"]
    assert valu["normalized_cycles_per_op"]["median"] == pytest.approx(8.0, abs=2.0)
    assert valu["isa_sanity"]["checks"]["clock64"]

    mfma_dep = benches["mfma.dependent_acc_32x32x32"]
    assert mfma_dep["ok"]
    assert mfma_dep["normalized_cycles_per_op"]["median"] == pytest.approx(18.0, abs=3.0)
    assert mfma_dep["isa_sanity"]["checks"]["mfma"]

    mfma_issue = benches["mfma.independent_acc_32x32x32_x4"]
    assert mfma_issue["ok"]
    assert mfma_issue["normalized_cycles_per_op"]["median"] == pytest.approx(17.0, abs=3.0)

    lds = benches["lds.dependent_gather_chase_i32"]
    assert lds["ok"]
    assert lds["correctness_ok"]
    assert lds["check_values"] == [lds["expected_check"]] * 7
    assert lds["normalized_cycles_per_op"]["median"] == pytest.approx(69.0, abs=8.0)
    assert lds["isa_sanity"]["checks"]["ttgir_local_gather"]
    assert lds["isa_sanity"]["checks"]["timed_ds_read_b32"]
    assert lds["isa_sanity"]["checks"]["no_timed_ds_write"]

    lds_issue = benches["lds.independent_gather_i32_x4"]
    assert lds_issue["ok"]
    assert lds_issue["correctness_ok"]
    assert lds_issue["normalized_cycles_per_op"]["median"] == pytest.approx(28.0, abs=8.0)

    global_load = benches["global.tl_load_dependent"]
    assert global_load["ok"]
    assert 100.0 < global_load["normalized_cycles_per_op"]["median"] < 200.0
    assert global_load["isa_sanity"]["checks"]["global_or_flat_load"]


@pytest.mark.skipif(not amd_latency.is_gfx950(), reason="requires active gfx950 HIP device")
def test_gfx950_tiny_global_distinguishes_load_paths():
    result = _run_or_skip_stale_clock64(bench="global", niter=2, warmup=1, reps=1)
    benches = {bench["name"]: bench for bench in result["benchmarks"]}
    assert "global.tl_load_dependent" in benches
    assert "global.direct_to_lds_composite_32x32" in benches

    pointer_chase = benches["global.tl_load_dependent"]
    assert pointer_chase["ok"]
    assert pointer_chase["stats"]["median"] > 0
    assert pointer_chase["net_ticks_per_op"] is None

    direct_to_lds = benches["global.direct_to_lds_composite_32x32"]
    if not direct_to_lds["ok"]:
        assert direct_to_lds["unsupported"]
        assert "unrealized_conversion_cast" in direct_to_lds["error"]
    else:
        assert direct_to_lds["stats"]["median"] > 0
        assert direct_to_lds["net_ticks_per_op"] is None
