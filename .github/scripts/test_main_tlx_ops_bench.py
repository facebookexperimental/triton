"""Main-push selection and regression reporting contracts."""
import copy
from pathlib import Path
import subprocess

import pytest

from main_tlx_ops_bench import Catalog
from main_tlx_ops_regressions import comment_regressions, compatible_history, detect_push
from test_tlx_ops_bench_regressions import history, night
from tlx_ops_bench_regressions import detect_all

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def catalog():
    return Catalog(ROOT)


def test_varlen_frontends_only_select_architectures_with_matching_shapes(catalog):
    benches = catalog.benchmarks()
    assert benches["bench_flash_attn_varlen"]["arches"] == {"gfx950"}
    assert benches["bench_flash_attn_varlen_ragged"]["arches"] == {"sm90"}


def test_historical_measurements_require_matching_harness_and_ancestral_source(tmp_path):

    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    def commit(path, content):
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content)
        git("add", path)
        git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
        return git("rev-parse", "HEAD")

    git("init", "-q")
    base = commit("python/test/tlx_benchmark/bench_mm.py", "initial harness\n")
    target = commit("kernel.py", "kernel change\n")
    old = night("20261101")
    old["sha"] = base
    current = night("20261104")
    current["sha"] = target
    plan = {
        "bench_ops": {"bench_mm": "flash_attn_mxfp8"}, "instruments":
        {"bench_mm": ["third_party/tlx/ops/kernels/mm/_shapes.py"]}
    }
    assert compatible_history([old], current, tmp_path, plan)[0]["ops"] == old["ops"]
    shape_change = commit("third_party/tlx/ops/kernels/mm/_shapes.py", "changed FLOP count\n")
    current["sha"] = shape_change
    assert compatible_history([old], current, tmp_path, plan)[0]["ops"] == {}
    new_harness = commit("python/test/tlx_benchmark/bench_mm.py", "changed harness\n")
    current["sha"] = new_harness
    assert compatible_history([old], current, tmp_path, plan)[0]["ops"] == {}
    old["sha"] = new_harness
    current["sha"] = target
    assert compatible_history([old], current, tmp_path, plan) == []


def test_shared_gemm_changes_reach_every_catalog_caller(catalog):
    plans = catalog.plan(["third_party/tlx/ops/kernels/mm/gfx950.py"])
    assert [plan["platform"] for plan in plans] == ["mi350"]
    assert set(plans[0]["modules"]) == {"bench_mm", "bench_addmm", "bench_moe"}
    assert {"mm", "addmm", "moe"} <= set(plans[0]["ops"])


def test_independent_ops_in_one_package_do_not_expand_to_siblings(catalog):
    plans = catalog.plan(["third_party/tlx/ops/kernels/kda/gfx950_prefill.py"])
    assert plans[0]["modules"] == {"bench_kda_prefill": ["kda_paged_prefill"]}
    assert [plan["platform"] for plan in plans] == ["mi350"]


def test_backend_selection_follows_reachable_implementations(catalog):
    plans = catalog.plan(["third_party/tlx/ops/kernels/flash_attn/sm100.py"])
    assert [plan["platform"] for plan in plans] == ["b200"]
    assert set(plans[0]["modules"]) == {"bench_flash_attn"}
    assert catalog.plan(["third_party/tlx/ops/kernels/flash_attn/gfx1250_bwd.py"]) == []


def test_normalization_family_reaches_the_frontend_that_measures_it(catalog):
    plans = catalog.plan(["third_party/tlx/ops/kernels/norm/_fp4.py"])
    assert [plan["platform"] for plan in plans] == ["b200"]
    assert set(plans[0]["modules"]) == {"bench_norm"}
    assert "layernorm_nvfp4_quantize" in plans[0]["modules"]["bench_norm"]


@pytest.mark.parametrize("path", [
    "third_party/tlx/ops/_catalog.py",
    "third_party/tlx/ops/__init__.py",
    "third_party/tlx/ops/kernels/mm/_shapes.py",
    "python/test/tlx_benchmark/_harness/driver.py",
    ".github/workflows/build.yml",
    "lib/Dialect/Triton/Transforms/Combine.cpp",
])
def test_nonimplementation_changes_do_not_allocate_benchmarks(catalog, path):
    assert catalog.plan([path]) == []


@pytest.mark.parametrize("value,expected", [(30, 1), (70, 1), (70.01, 0), (85, 0), (100, 0)])
def test_push_reporting_uses_the_shared_detector_evidence_thresholds(value, expected):
    current = night("20261104", value)
    alerts, compared = detect_push(history(), current)
    assert alerts == [a for a in detect_all(history() + [current]) if a["sha"] == current["sha"]]
    assert len(alerts) == expected
    assert compared == {"flash_attn_mxfp8": 3}


def test_daily_history_cannot_overwrite_the_push_or_supply_future_evidence():
    current = night("20261104", 30)
    same_day = night("20261104", 200)
    same_day["sha"] = "a" * 40
    future = night("20261105", 200)
    assert detect_push(history() + [same_day, future], current) == detect_push(history(), current)
    assert detect_push(history()[:2], current) == ([], {})


def test_regression_comment_is_relevant_and_idempotent():
    alert = detect_all(history() + [night("20261104", 30)])[0]
    comments, calls = [], []

    def api(method, path, fields=None, paginate=False):
        calls.append((method, path, fields))
        if method == "GET":
            return comments
        comments.append(copy.deepcopy(fields))
        return {"html_url": "https://github.com/o/r/pull/1#issuecomment-10"}

    prs = {1: {"flash_attn_mxfp8"}, 2: {"mm"}}
    assert comment_regressions([alert], prs, "o/r", "https://github.com/o/r/actions/runs/10", alert["sha"], api)
    assert comment_regressions([alert], prs, "o/r", "https://github.com/o/r/actions/runs/10", alert["sha"], api) == []
    assert len(comments) == 1
    assert all("/issues/1/comments" in path for _, path, _ in calls)
    assert "does not establish this PR as the cause" in comments[0]["body"]


def test_normalization_comments_keep_case_level_public_op_relevance():
    alert = detect_all(history() + [night("20261104", 30)])[0]
    alert["op"] = "norm_fwd"
    for case in alert["cases"]:
        case["params"] = {"input": "softmax:softmax:128x65536:default:q1:j0"}
    comments = []

    def api(method, path, fields=None, paginate=False):
        if method == "GET":
            return []
        comments.append(fields["body"])
        return {"html_url": "https://github.com/o/r/pull/1#issuecomment-10"}

    prs = {1: {"layernorm_nvfp4_quantize"}}
    assert comment_regressions([alert], prs, "o/r", "run", alert["sha"], api) == []
    assert comments == []
    alert["cases"][0]["params"] = {"input": "nvfp4:layer:8192x384:default:q16:j0"}
    assert comment_regressions([alert], prs, "o/r", "run", alert["sha"], api)
    assert "nvfp4:layer" in comments[0] and "softmax" not in comments[0]


def test_no_regression_never_contacts_github():

    def api(*args, **kwargs):
        pytest.fail("No regression must not write or read PR comments")

    assert comment_regressions([], {1: {"mm"}}, "o/r", "run", "a" * 40, api) == []
