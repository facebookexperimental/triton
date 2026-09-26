#!/usr/bin/env python3

"""Run a GEO Kernel Optimizer workspace without Buck.

The adapter preserves GEO's workspace-local test, benchmark, and repro drivers,
but materializes their Buck ``base_module`` mappings as an ordinary Python
package.  Run it with the Python environment belonging to the OSS Triton tree.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


_DEFAULT_OSS_ROOT = Path(__file__).resolve().parents[4]
_WORKSPACE_TARGETS = {"test", "bench", "repro"}


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return payload


def _package_name(meta: dict[str, object]) -> str:
    module_name = str(meta.get("module_dotted_path") or "")
    package = module_name.partition(".")[0]
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", package):
        raise RuntimeError(f"invalid workspace package in module path: {module_name}")
    return package


def _write_package_init(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    init = directory / "__init__.py"
    if not init.exists():
        init.write_text("")


def _rewrite_source(text: str, *, package: str, workspace_name: str) -> str:
    # Workspaces may live below a directory containing a dash (for example
    # ``post-fuser``).  Such a path is valid for Buck's source mapping but not
    # in a Python import statement.  Normalize generated imports to the stable
    # package already recorded in kernel_meta.json.
    invalid_prefix = f"triton.tools.post-fuser.optimizer_workspaces.{workspace_name}"
    text = text.replace(invalid_prefix, package)
    text = text.replace(f"{package}.runtime.geo_harness", f"{package}.runtime.harness")
    text = text.replace(
        "from gem.next_gen.geo.common.testing.standards import",
        f"from {package}.runtime.harness.standards import",
    )
    text = text.replace(
        "from gem.next_gen.geo.common.kernel_perf_store import stable_hash",
        f"from {package}.runtime.harness.oss_compat import stable_hash",
    )
    return text


def _copy_python(
    source: Path,
    target: Path,
    *,
    package: str,
    workspace_name: str,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        _rewrite_source(
            source.read_text(), package=package, workspace_name=workspace_name
        )
    )


def materialize_workspace(workspace: Path) -> tuple[Path, dict[str, object]]:
    workspace = workspace.resolve()
    meta = _read_json(workspace / "inputs" / "kernel_meta.json")
    package = _package_name(meta)
    python_root = workspace / ".oss_adapter" / "python"
    package_root = python_root / package
    if package_root.exists():
        shutil.rmtree(package_root)

    for directory in (
        package_root,
        package_root / "kernel",
        package_root / "tests",
        package_root / "runtime",
        package_root / "runtime" / "harness",
    ):
        _write_package_init(directory)

    module_leaf = str(meta["module_dotted_path"]).rsplit(".", 1)[-1]
    _copy_python(
        workspace / "kernels" / "current.py",
        package_root / "kernel" / f"{module_leaf}.py",
        package=package,
        workspace_name=workspace.name,
    )

    for path_key, module_key in (
        ("test_path_rel", "test_module_dotted_path"),
        ("bench_path_rel", "bench_module_dotted_path"),
    ):
        source = workspace / str(meta[path_key])
        module_leaf = str(meta[module_key]).rsplit(".", 1)[-1]
        _copy_python(
            source,
            package_root / "tests" / f"{module_leaf}.py",
            package=package,
            workspace_name=workspace.name,
        )

    _copy_python(
        workspace / "runtime" / "workspace_drivers.py",
        package_root / "runtime" / "workspace_drivers.py",
        package=package,
        workspace_name=workspace.name,
    )
    for source in sorted((workspace / "runtime" / "harness").glob("*.py")):
        _copy_python(
            source,
            package_root / "runtime" / "harness" / source.name,
            package=package,
            workspace_name=workspace.name,
        )
    standards = package_root / "runtime" / "harness" / "standards.py"
    if not standards.exists():
        standards.write_text(
            "ACCURACY_ATOL_BF16 = 1e-2\n"
            "ACCURACY_RTOL_BF16 = 1e-2\n"
            "ACCURACY_ATOL_FP32 = 1e-3\n"
            "ACCURACY_RTOL_FP32 = 1e-3\n"
            "BENCH_EVAL_WARMUP = 200\n"
            "BENCH_EVAL_REP = 200\n"
            "BENCH_SEARCH_WARMUP = 25\n"
            "BENCH_SEARCH_REP = 100\n"
        )
    (package_root / "runtime" / "harness" / "oss_compat.py").write_text(
        "import hashlib\n"
        "import json\n\n"
        "def stable_hash(value):\n"
        "    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), "
        "default=str).encode()\n"
        "    return hashlib.sha256(encoded).hexdigest()\n"
    )
    return python_root, meta


def _configure_oss_imports(oss_root: Path, python_root: Path) -> None:
    for path in (python_root, oss_root / "python", oss_root):
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def _run_driver(
    command: str,
    workspace: Path,
    forwarded: list[str],
    *,
    oss_root: Path,
) -> None:
    python_root, meta = materialize_workspace(workspace)
    package = _package_name(meta)
    _configure_oss_imports(oss_root, python_root)
    os.chdir(oss_root)
    module = importlib.import_module(f"{package}.runtime.workspace_drivers")
    entry: Callable[[], None] = getattr(module, f"{command}_main")
    sys.argv = [f"geo-oss-{command}", "--workspace", str(workspace), *forwarded]
    entry()


def _oss_env(env: dict[str, str], oss_root: Path) -> dict[str, str]:
    result = dict(env)
    prefixes = [str((oss_root / "python").resolve()), str(oss_root.resolve())]
    existing = result.get("PYTHONPATH", "")
    if existing:
        prefixes.append(existing)
    result["PYTHONPATH"] = os.pathsep.join(prefixes)
    return result


def _run_oss_capture(
    *,
    cmd_args: list[str],
    env: dict[str, str],
    cwd: str,
    timeout_s: float,
    live_log_path: Path | None = None,
    live_log_append: bool = False,
    oss_root: Path,
) -> subprocess.CompletedProcess[str]:
    del cwd
    try:
        separator = cmd_args.index("--")
    except ValueError as exc:
        raise RuntimeError(
            f"workspace command has no '--' separator: {cmd_args}"
        ) from exc
    target = cmd_args[separator - 1]
    command = target.rsplit(":", 1)[-1]
    if command not in {"test", "bench", "repro"}:
        raise RuntimeError(f"unsupported OSS workspace target: {target}")
    workspace_arg = cmd_args[separator + 1 :]
    adapter_args = [
        os.environ.get("GEO_OSS_PYTHON", sys.executable),
        str(Path(__file__).resolve()),
        command,
        "--oss-root",
        str(oss_root.resolve()),
        *workspace_arg,
    ]
    completed = subprocess.run(
        adapter_args,
        cwd=oss_root,
        env=_oss_env(env, oss_root),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if live_log_path is not None:
        live_log_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if live_log_append else "w"
        with live_log_path.open(mode) as output:
            output.write(completed.stdout or "")
        with live_log_path.with_name(live_log_path.name + ".err").open(mode) as error:
            error.write(completed.stderr or "")
    return completed


def _workspace_target(args: list[str], *, fbcode_root: Path) -> tuple[Path, str] | None:
    for arg in args:
        if arg.startswith("fbcode//"):
            target_text = arg[len("fbcode//") :]
        elif arg.startswith("//"):
            target_text = arg[2:]
        else:
            continue
        if ":" not in target_text:
            continue
        path_text, command = target_text.rsplit(":", 1)
        if command not in _WORKSPACE_TARGETS:
            continue
        workspace = (fbcode_root / path_text).resolve()
        if (workspace / "inputs" / "kernel_meta.json").is_file():
            return workspace, command
    return None


def _emit_repro_launcher(
    *,
    workspace: Path,
    command: str,
    forwarded: list[str],
    oss_root: Path,
    fbcode_root: Path,
) -> None:
    output = (
        fbcode_root.parent
        / "buck-out"
        / "geo_oss_adapter"
        / workspace.name
        / f"{command}.par"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    python = os.environ.get("GEO_OSS_PYTHON", sys.executable)
    script = Path(__file__).resolve()
    output.write_text(
        "#!/bin/sh\n"
        f"exec {shlex.quote(python)} {shlex.quote(str(script))} "
        f'{shlex.quote(command)} --oss-root {shlex.quote(str(oss_root))} "$@"\n'
    )
    output.chmod(0o755)
    print(
        " ".join([shlex.quote(str(output)), *(shlex.quote(arg) for arg in forwarded)])
    )


def _run_buck2_compat(args: list[str], *, oss_root: Path, fbcode_root: Path) -> None:
    action = next((arg for arg in args if arg in {"build", "run"}), "")
    forwarded = args[args.index("--") + 1 :] if "--" in args else []
    is_setup_tool = any(arg.endswith(":kernel_optimizer_setup_tool") for arg in args)
    if is_setup_tool and action == "run" and forwarded[:1] == ["validate"]:
        validate_parser = argparse.ArgumentParser(add_help=False)
        validate_parser.add_argument("--workspace", required=True, type=Path)
        validate_parser.add_argument("--fbcode-root", required=True, type=Path)
        validate_parser.add_argument("--gpu", required=True, type=int)
        validate_parser.add_argument("--timeout-s", type=float, default=3600.0)
        validate_args = validate_parser.parse_args(forwarded[1:])
        _run_validate(
            validate_args.workspace.resolve(),
            oss_root=oss_root,
            fbcode_root=validate_args.fbcode_root.resolve(),
            gpu=validate_args.gpu,
            timeout_s=validate_args.timeout_s,
        )
        return

    target = _workspace_target(args, fbcode_root=fbcode_root)
    if target is None:
        real_buck2 = os.environ.get("GEO_OSS_REAL_BUCK2", "/usr/local/bin/buck2")
        os.execv(real_buck2, [real_buck2, *args])

    workspace, command = target
    if action == "build":
        materialize_workspace(workspace)
        print(f"GEO OSS adapter materialized {workspace}")
        return
    if action != "run":
        raise RuntimeError(f"unsupported buck2 action for GEO OSS target: {action}")
    if "--emit-shell" in args:
        _emit_repro_launcher(
            workspace=workspace,
            command=command,
            forwarded=forwarded,
            oss_root=oss_root,
            fbcode_root=fbcode_root,
        )
        return

    completed = subprocess.run(
        [
            os.environ.get("GEO_OSS_PYTHON", sys.executable),
            str(Path(__file__).resolve()),
            command,
            "--workspace",
            str(workspace),
            "--oss-root",
            str(oss_root),
            *forwarded,
        ],
        cwd=oss_root,
        env=_oss_env(os.environ, oss_root),
        check=False,
    )
    raise SystemExit(completed.returncode)


def _run_validate(
    workspace: Path,
    *,
    oss_root: Path,
    fbcode_root: Path,
    gpu: int,
    timeout_s: float,
) -> None:
    fbcode_root = fbcode_root.resolve()
    if str(fbcode_root) not in sys.path:
        sys.path.insert(0, str(fbcode_root))
    _configure_oss_imports(oss_root, workspace / ".oss_adapter" / "python")

    from gem.next_gen.geo.agents.kernel_optimizer import setup_tool
    from gem.next_gen.geo.agents.kernel_optimizer.utils import measurement, profiling

    def capture(**kwargs: Any) -> subprocess.CompletedProcess[str]:
        return _run_oss_capture(**kwargs, oss_root=oss_root)

    def resolve_repro_launch_command(**kwargs: Any) -> tuple[str, list[str]]:
        profile_target = str(kwargs.get("profile_target") or "fused")
        forwarded = [
            str(Path(__file__).resolve()),
            "repro",
            "--oss-root",
            str(oss_root.resolve()),
            "--workspace",
            str(Path(kwargs["workspace"]).resolve()),
            "--shape",
            ",".join(str(dim) for dim in kwargs["shape"]),
            "--launch-count",
            str(max(1, int(kwargs["launch_count"]))),
        ]
        if profile_target != "fused":
            forwarded.extend(["--profile-target", profile_target])
        return sys.executable, forwarded

    measurement.run_buck2_capture = capture
    setup_tool.run_buck2_capture = capture
    profiling._resolve_repro_launch_command = resolve_repro_launch_command
    result = setup_tool.validate_setup(
        workspace=workspace,
        fbcode_root=fbcode_root,
        gpu_id=gpu,
        timeout_s=timeout_s,
    )
    print(json.dumps(result))


def main() -> None:
    if sys.argv[1:2] == ["buck2"]:
        fbcode_root_text = os.environ.get("GEO_OSS_FBCODE_ROOT")
        if not fbcode_root_text:
            raise RuntimeError(
                "GEO_OSS_FBCODE_ROOT is required for buck2 compatibility"
            )
        _run_buck2_compat(
            sys.argv[2:],
            oss_root=_DEFAULT_OSS_ROOT.resolve(),
            fbcode_root=Path(fbcode_root_text).resolve(),
        )
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("test", "bench", "repro", "validate"))
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--oss-root", type=Path, default=_DEFAULT_OSS_ROOT)
    args, forwarded = parser.parse_known_args()
    if args.workspace is None:
        parser.error("--workspace is required")
    if args.command == "validate":
        validate_parser = argparse.ArgumentParser(add_help=False)
        validate_parser.add_argument("--fbcode-root", required=True, type=Path)
        validate_parser.add_argument("--gpu", required=True, type=int)
        validate_parser.add_argument("--timeout-s", type=float, default=3600.0)
        validate_args = validate_parser.parse_args(forwarded)
        _run_validate(
            args.workspace.resolve(),
            oss_root=args.oss_root.resolve(),
            fbcode_root=validate_args.fbcode_root,
            gpu=validate_args.gpu,
            timeout_s=validate_args.timeout_s,
        )
        return
    _run_driver(
        args.command,
        args.workspace.resolve(),
        forwarded,
        oss_root=args.oss_root.resolve(),
    )


if __name__ == "__main__":
    main()
