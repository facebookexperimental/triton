import os
from pathlib import Path

import pytest

from select_tests import GPUS, SELF_TESTS, TEST_MARKERS, main, named_gpus, scan_list, select

FA = "third_party/tlx/ops/kernels/flash_attn"
FA_SM100_TEST = "python/test/unit/tlx_ops/test_flash_attn_sm100.py"
FA_SM90_TEST = "python/test/unit/tlx_ops/test_flash_attn_sm90.py"
NOTHING = {"b200": [], "h100": []}


# A kernel change is checked on each NVIDIA GPU it targets and nowhere else.
@pytest.mark.parametrize("kernel, b200, h100", [
    (f"{FA}/sm100.py", [FA_SM100_TEST], []),
    (f"{FA}/sm90.py", [], [FA_SM90_TEST]),
    (f"{FA}/_shapes.py", [FA_SM100_TEST], [FA_SM90_TEST]),
    (f"{FA}/gfx950.py", [], []),
])
def test_kernel_change_is_checked_on_the_gpus_it_targets(kernel, b200, h100):
    assert select([kernel]) == {"b200": b200, "h100": h100}


def test_pipeline_change_runs_the_self_test_on_each_gpu():
    tests = select([".github/workflows/cutracer.yml", ".github/scripts/cutracer_ci/findings.py"])
    assert tests == {gpu: [test] for gpu, test in SELF_TESTS.items()}


@pytest.mark.parametrize("changed", [["lib/Dialect/Triton/Transforms/Combine.cpp"], [".github/workflows/b200.yml"]])
def test_other_changes_check_nothing(changed):
    assert select(changed) == NOTHING


def test_pipeline_and_kernel_changes_run_both():
    tests = select([".github/workflows/cutracer.yml", f"{FA}/sm100.py"])
    assert tests["b200"] == [SELF_TESTS["b200"], FA_SM100_TEST]
    assert tests["h100"] == [SELF_TESTS["h100"]]


def test_pipeline_change_with_a_compiler_change_still_runs_the_self_test():
    tests = select([".github/actions/setup-cutracer/action.yml", "lib/Dialect/Triton/Transforms/Combine.cpp"])
    assert tests == {gpu: [test] for gpu, test in SELF_TESTS.items()}


def test_manual_run_checks_the_named_test_on_the_named_gpu(tmp_path, monkeypatch):
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    main(["--test", FA_SM90_TEST, "--gpu", "h100"])
    assert output.read_text() == f'b200=[]\nh100=["{FA_SM90_TEST}"]\n'


def test_pull_request_reads_the_changed_files(tmp_path, monkeypatch):
    changed = tmp_path / "changed.txt"
    changed.write_text("README.md\n")
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    main(["--changed-files", str(changed)])
    assert output.read_text() == "b200=[]\nh100=[]\n"


# A renamed or moved test must not drop out of the scan unnoticed, and a scan
# list must not hold a test for the other GPU.
@pytest.mark.parametrize("gpu", GPUS)
def test_scan_list_names_existing_tests_for_its_gpu(gpu):
    repo = Path(__file__).resolve().parents[3]
    for test in scan_list(gpu):
        path, *names = test.split("[")[0].split("::")
        assert (repo / path).is_file(), test
        source = (repo / path).read_text()
        for name in names:
            assert f"def {name}(" in source or f"class {name}" in source, test
        named = named_gpus(os.path.basename(path), TEST_MARKERS)
        assert not named or gpu in named, test
