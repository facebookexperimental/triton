from __future__ import annotations

import pathlib
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from inductor_fusions import ARCH_PACKAGES, cases_for_arch, catalog  # noqa: E402
from inductor_fusions.common import CodeContract  # noqa: E402
from inductor_fusions.compat import (  # noqa: E402
    CompatibilityError, require_include_fallback,
)
from inductor_fusions.measure import (  # noqa: E402
    PairedSample, summarize_pairs,
)


def test_catalog_has_stable_architecture_boundaries_and_enabled_mi350_cases():
    assert ARCH_PACKAGES == {
        "sm90": "h100",
        "sm100": "b200",
        "gfx942": "mi300",
        "gfx950": "mi350",
    }
    assert set(catalog()) == set(ARCH_PACKAGES)
    assert [case.name for case in cases_for_arch("gfx950")] == ["gfx950_01_double_layernorm"]
    assert cases_for_arch("unknown") == ()


def test_authoritative_mi350_problems_are_visible_without_allocating_inputs():
    problems = {case.name: case.problem for case in cases_for_arch("gfx950")}
    assert problems == {"gfx950_01_double_layernorm": "M=1024 N=6144 dtype=fp16"}


def test_summary_uses_median_of_paired_speedups_and_marks_noise():
    stable = summarize_pairs(
        [PairedSample(10.0, 5.0, "AB"), PairedSample(12.0, 6.0, "BA")],
        max_spread=0.01,
    )
    assert stable.speedup == 2.0
    assert not stable.noisy

    noisy = summarize_pairs(
        [
            PairedSample(10.0, 5.0, "AB"),
            PairedSample(10.0, 8.0, "BA"),
            PairedSample(10.0, 4.0, "AB"),
            PairedSample(10.0, 7.0, "BA"),
        ],
        max_spread=0.05,
    )
    assert noisy.noisy
    assert noisy.to_dict()["status"] == "noisy"


def test_codegen_contract_rejects_wrong_variant_and_launch_count():
    contract = CodeContract(
        required_after=("fused_marker", ),
        forbidden_after=("global_intermediate", ),
        expected_after_launches=2,
    )
    contract.validate("baseline", "fused_marker\na.run(\nb.run(")
    with pytest.raises(AssertionError, match="baseline unexpectedly"):
        contract.validate("fused_marker", "fused_marker\na.run(\nb.run(")
    with pytest.raises(AssertionError, match="produced 1 launches"):
        contract.validate("baseline", "fused_marker\na.run(")


def test_compatibility_check_requires_include_fallback():

    def compatible(*, include_fallback=True):
        pass

    def incompatible():
        pass

    require_include_fallback(compatible)
    with pytest.raises(CompatibilityError, match="include_fallback"):
        require_include_fallback(incompatible)


def test_list_path_does_not_load_the_runtime(monkeypatch, capsys):
    import run_torchtlx_fusions as runner

    def should_not_run(*args, **kwargs):
        raise AssertionError("runtime loaded during --list")

    monkeypatch.setattr(runner, "install_torchtlx", should_not_run)
    assert runner.main(["--list"]) == 0
    output = capsys.readouterr().out
    assert "gfx950:" in output
    assert "gfx950_01_double_layernorm" in output


def test_arch_prefixed_case_id_selects_exactly_one_case():
    import run_torchtlx_fusions as runner

    selected = runner._select_cases("gfx950", ["gfx950_01_double_layernorm"])
    assert [case.name for case in selected] == ["gfx950_01_double_layernorm"]


def test_default_run_is_the_forced_comparison_only():
    import run_torchtlx_fusions as runner

    args = runner._parser().parse_args(["--case", "gfx950_01_double_layernorm"])
    assert args.selection == "forced"
    assert args.samples == 5


def test_double_layernorm_exposes_forced_and_production_variants():
    case = cases_for_arch("gfx950")[0]
    assert case.variant("forced", "before") == (None, {"triton.multi_kernel": 2})
    assert case.variant("forced", "after") == ("allow", {"triton.multi_kernel": 3})
    assert case.variant("autotuned", "before") == (None, {"triton.multi_kernel": 1})
    assert case.variant("autotuned", "after") == ("allow", {"triton.multi_kernel": 1})


def test_autotuned_codegen_classifies_candidate_or_fallback():
    contract = CodeContract(required_after=("candidate_a", "candidate_b"))
    assert contract.classify_autotuned("candidate_a candidate_b") == "candidate_present"
    assert contract.classify_autotuned("ordinary kernel") == "fallback_selected"
    with pytest.raises(AssertionError, match="only part"):
        contract.classify_autotuned("candidate_a")


def test_process_batches_are_reassembled_as_paired_abba_samples():
    import run_torchtlx_fusions as runner

    samples = runner._pair_abba_blocks(
        before_ab=(10.0, 11.0),
        after_ab=(5.0, 5.5),
        after_ba=(6.0, 6.5),
        before_ba=(12.0, 13.0),
    )
    assert [(sample.order, sample.before_us, sample.after_us) for sample in samples] == [
        ("AB", 10.0, 5.0),
        ("BA", 12.0, 6.0),
        ("AB", 11.0, 5.5),
        ("BA", 13.0, 6.5),
    ]


def test_reference_process_batches_are_paired_by_sample_index():
    import run_torchtlx_fusions as runner

    samples = runner._pair_reference_blocks(before=(10.0, 12.0), after=(5.0, 6.0))
    assert [(sample.order, sample.before_us, sample.after_us) for sample in samples] == [
        ("AB", 10.0, 5.0),
        ("AB", 12.0, 6.0),
    ]
    with pytest.raises(ValueError, match="different lengths"):
        runner._pair_reference_blocks(before=(10.0, ), after=(5.0, 6.0))


def test_fast_path_uses_two_timing_workers_and_one_candidate_validation(monkeypatch):
    import run_torchtlx_fusions as runner

    calls = []

    def run_worker(case, **kwargs):
        calls.append((kwargs["side"], kwargs.get("validate", False), kwargs["samples"]))
        before = kwargs["side"] == "before"
        return {
            "generated": "baseline" if before else "candidate",
            "torch": "test",
            "triton": "test",
            "device": "test",
            "multi_kernel_choice": None,
            "latencies_us": [10.0, 10.0] if before else [5.0, 5.0],
        }

    monkeypatch.setattr(runner, "_assert_device_idle", lambda _: None)
    monkeypatch.setattr(runner, "_run_worker_batch", run_worker)
    summary, _ = runner._run_reference_comparison(
        cases_for_arch("gfx950")[0],
        arch="gfx950",
        warmup=1,
        rep=1,
        samples=2,
        max_spread=0.05,
        device_index=0,
    )
    assert summary.speedup == 2.0
    assert calls == [("before", False, 2), ("after", False, 2), ("after", True, 0)]
