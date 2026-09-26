from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import third_party.tlx.tools.agents.geo_oss_adapter as adapter
from third_party.tlx.tools.agents.geo_oss_adapter import (
    _run_buck2_compat,
    _workspace_target,
    materialize_workspace,
)


def test_materialize_workspace_rewrites_buck_only_modules(tmp_path: Path) -> None:
    workspace = tmp_path / "post-fuser" / "optimizer_workspaces" / "case"
    (workspace / "inputs").mkdir(parents=True)
    (workspace / "kernels").mkdir()
    (workspace / "tests").mkdir()
    (workspace / "runtime" / "harness").mkdir(parents=True)
    (workspace / "inputs" / "kernel_meta.json").write_text(
        json.dumps(
            {
                "module_dotted_path": "geo_ws_case.kernel.candidate",
                "test_module_dotted_path": "geo_ws_case.tests.test_candidate",
                "bench_module_dotted_path": "geo_ws_case.tests.bench_candidate",
                "test_path_rel": "tests/test_candidate.py",
                "bench_path_rel": "tests/bench_candidate.py",
            }
        )
    )
    (workspace / "kernels" / "current.py").write_text("VALUE = 7\n")
    invalid = "triton.tools.post-fuser.optimizer_workspaces.case"
    (workspace / "tests" / "test_candidate.py").write_text(
        "from geo_ws_case.kernel.candidate import VALUE\n"
    )
    (workspace / "tests" / "bench_candidate.py").write_text("BENCH = True\n")
    (workspace / "runtime" / "workspace_drivers.py").write_text("DRIVER = True\n")
    (workspace / "runtime" / "harness" / "base.py").write_text(
        f"from {invalid}.runtime.harness.config_codec import TOKEN\n"
        "from gem.next_gen.geo.common.testing.standards import BENCH_EVAL_REP\n"
    )
    (workspace / "runtime" / "harness" / "config_codec.py").write_text(
        "from gem.next_gen.geo.common.kernel_perf_store import stable_hash\n"
        "TOKEN = 11\n"
    )
    (workspace / "runtime" / "harness" / "standards.py").write_text(
        "BENCH_EVAL_REP = 200\n"
    )

    python_root, _ = materialize_workspace(workspace)
    sys.path.insert(0, str(python_root))
    try:
        candidate = importlib.import_module("geo_ws_case.kernel.candidate")
        base = importlib.import_module("geo_ws_case.runtime.harness.base")
        config_codec = importlib.import_module(
            "geo_ws_case.runtime.harness.config_codec"
        )
    finally:
        sys.path.remove(str(python_root))
        for name in list(sys.modules):
            if name == "geo_ws_case" or name.startswith("geo_ws_case."):
                del sys.modules[name]

    assert candidate.VALUE == 7
    assert base.TOKEN == 11
    assert base.BENCH_EVAL_REP == 200
    assert len(config_codec.stable_hash({"value": 1})) == 64


def test_buck2_build_materializes_workspace(tmp_path: Path) -> None:
    fbcode_root = tmp_path / "fbcode"
    workspace = fbcode_root / "optimizer_workspaces" / "case"
    (workspace / "inputs").mkdir(parents=True)
    (workspace / "kernels").mkdir()
    (workspace / "tests").mkdir()
    (workspace / "runtime" / "harness").mkdir(parents=True)
    (workspace / "inputs" / "kernel_meta.json").write_text(
        json.dumps(
            {
                "module_dotted_path": "geo_ws_case.kernel.candidate",
                "test_module_dotted_path": "geo_ws_case.tests.test_candidate",
                "bench_module_dotted_path": "geo_ws_case.tests.bench_candidate",
                "test_path_rel": "tests/test_candidate.py",
                "bench_path_rel": "tests/bench_candidate.py",
            }
        )
    )
    (workspace / "kernels" / "current.py").write_text("VALUE = 7\n")
    (workspace / "tests" / "test_candidate.py").write_text("TEST = True\n")
    (workspace / "tests" / "bench_candidate.py").write_text("BENCH = True\n")
    (workspace / "runtime" / "workspace_drivers.py").write_text("DRIVER = True\n")

    _run_buck2_compat(
        ["build", "//optimizer_workspaces/case:test"],
        oss_root=tmp_path,
        fbcode_root=fbcode_root,
    )

    assert (workspace / ".oss_adapter/python/geo_ws_case/kernel/candidate.py").is_file()


def test_workspace_target_ignores_setup_tool_target(tmp_path: Path) -> None:
    assert (
        _workspace_target(
            [
                "run",
                "fbcode//gem/next_gen/geo/agents/kernel_optimizer:kernel_optimizer_setup_tool",
                "--",
                "validate",
                "--workspace",
                "/tmp/workspace",
            ],
            fbcode_root=tmp_path,
        )
        is None
    )


def test_workspace_target_accepts_fbcode_prefix(tmp_path: Path) -> None:
    workspace = tmp_path / "optimizer_workspaces" / "case"
    (workspace / "inputs").mkdir(parents=True)
    (workspace / "inputs" / "kernel_meta.json").write_text("{}")

    assert _workspace_target(
        ["run", "fbcode//optimizer_workspaces/case:repro"],
        fbcode_root=tmp_path,
    ) == (workspace.resolve(), "repro")


def test_buck2_setup_validation_routes_to_oss(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    calls = []

    def fake_validate(*args, **kwargs) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(adapter, "_run_validate", fake_validate)
    _run_buck2_compat(
        [
            "run",
            "fbcode//gem/next_gen/geo/agents/kernel_optimizer:kernel_optimizer_setup_tool",
            "--",
            "validate",
            "--workspace",
            str(workspace),
            "--fbcode-root",
            str(tmp_path),
            "--gpu",
            "1",
            "--timeout-s",
            "42",
        ],
        oss_root=tmp_path,
        fbcode_root=tmp_path,
    )

    assert calls == [
        (
            (workspace.resolve(),),
            {
                "oss_root": tmp_path,
                "fbcode_root": tmp_path,
                "gpu": 1,
                "timeout_s": 42.0,
            },
        )
    ]
