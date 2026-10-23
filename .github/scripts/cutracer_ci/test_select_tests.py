import json

import pytest

from select_tests import SELF_TEST, main, select

FA_SM100 = "third_party/tlx/ops/kernels/flash_attn/sm100.py"


def test_kernel_change_checks_its_b200_tests():
    tests = select([FA_SM100])
    assert "python/test/unit/tlx_ops/test_flash_attn_sm100.py" in tests
    assert all("sm100" in test for test in tests)


def test_pipeline_change_runs_the_self_test():
    assert select([".github/workflows/cutracer.yml", ".github/scripts/cutracer_ci/findings.py"]) == [SELF_TEST]


@pytest.mark.parametrize("changed", [["lib/Dialect/Triton/Transforms/Combine.cpp"], [".github/workflows/b200.yml"],
                                     ["third_party/tlx/ops/kernels/flash_attn/gfx950.py"]])
def test_other_changes_check_nothing(changed):
    assert select(changed) == []


def test_pipeline_and_kernel_changes_run_both():
    tests = select([".github/workflows/cutracer.yml", FA_SM100])
    assert tests[0] == SELF_TEST
    assert "python/test/unit/tlx_ops/test_flash_attn_sm100.py" in tests


def test_pipeline_change_with_a_compiler_change_still_runs_the_self_test():
    assert select([".github/actions/setup-cutracer/action.yml",
                   "lib/Dialect/Triton/Transforms/Combine.cpp"]) == [SELF_TEST]


def test_manual_run_checks_the_named_test(tmp_path, monkeypatch, capsys):
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    main(["--test", "python/test/unit/tlx_ops/test_gemm_sm100.py"])
    assert output.read_text() == 'tests=["python/test/unit/tlx_ops/test_gemm_sm100.py"]\n'


def test_pull_request_reads_the_changed_files(tmp_path, monkeypatch):
    changed = tmp_path / "changed.txt"
    changed.write_text("README.md\n")
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    main(["--changed-files", str(changed)])
    assert json.loads(output.read_text().partition("=")[2]) == []
