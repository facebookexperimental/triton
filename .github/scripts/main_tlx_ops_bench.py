#!/usr/bin/env python3
"""Select main-push TLX benchmarks from catalog implementation dependencies."""
import argparse
import ast
from functools import cache
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
from types import SimpleNamespace

from tlx_ops_bench_bisect import bench_modules

ROOT = Path(__file__).resolve().parents[2]
KERNELS = "third_party/tlx/ops/kernels/"
PLATFORMS = {"h100": "sm90", "b200": "sm100", "mi350": "gfx950"}


def implementation(path):
    prefixes = (KERNELS, "third_party/tlx/tutorials/")
    return (any(path.startswith(prefix) and "/" in path[len(prefix):] for prefix in prefixes) and path.endswith(".py")
            and not path.endswith(("_shapes.py", "_reference.py", "/bench.py")))


class Catalog:

    def __init__(self, repo):
        self.repo = Path(repo)
        self.specs = runpy.run_path(str(self.repo / "third_party/tlx/ops/_catalog.py"))["CATALOG"]

    @cache
    def module_path(self, module):
        prefixes = {
            "triton.tlx.ops.kernels": "third_party/tlx/ops/kernels",
            "triton.tlx.tutorials": "third_party/tlx/tutorials",
        }
        for prefix, directory in prefixes.items():
            if module == prefix or module.startswith(prefix + "."):
                stem = self.repo / directory / module[len(prefix):].lstrip(".").replace(".", "/")
                for path in (stem.with_suffix(".py"), stem / "__init__.py"):
                    if path.is_file():
                        return path
        return None

    @cache
    def imports(self, module):
        path = self.module_path(module)
        if path is None:
            return ()
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        found = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = importlib.util.resolve_name("." * node.level + base, package)
                found.add(base)
                found.update(base + "." + alias.name for alias in node.names)
        return tuple(sorted(name for name in found if self.module_path(name) is not None))

    @cache
    def dependencies(self, module):
        visited, paths, pending = set(), set(), [module]
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            path = self.module_path(current)
            if path is None:
                continue
            paths.add(path.relative_to(self.repo).as_posix())
            for parent in path.parents:
                if not parent.is_relative_to(self.repo / KERNELS):
                    break
                init = parent / "__init__.py"
                if init.is_file():
                    paths.add(init.relative_to(self.repo).as_posix())
            pending.extend(self.imports(current))
        return frozenset(paths)

    def affected(self, paths):
        changed = {p for p in paths if implementation(p)}
        return {
            platform: {
                spec.op
                for spec in self.specs
                if spec.arch == arch and changed.intersection(self.dependencies(spec.impl.split(":", 1)[0]))
            }
            for platform, arch in PLATFORMS.items()
        }

    @cache
    def focus_arches(self, module):
        path = self.module_path(module)
        if path is None:
            return frozenset()
        values = {}
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if isinstance(node.value, ast.Dict):
                    for target in targets:
                        if isinstance(target, ast.Name):
                            try:
                                values[target.id] = ast.literal_eval(node.value)
                            except (ValueError, TypeError):
                                pass
                if any(isinstance(t, ast.Name) and t.id == "FOCUS" for t in targets):
                    registry = node.value
                    if isinstance(registry, ast.Call) and len(registry.args) >= 3:
                        defaults = registry.args[2]
                        defaults = values.get(defaults.id, {}) if isinstance(defaults,
                                                                             ast.Name) else ast.literal_eval(defaults)
                        return frozenset(arch for arch, suites in defaults.items() if suites)
        return frozenset()

    def frontend_arches(self, tree, shape_modules, arches):
        shape_filter = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "shapes"),
                            None)
        if shape_filter is None:
            return arches
        shared = runpy.run_path(str(self.repo / KERNELS / "_shape_suites.py"))
        namespace = {
            "__name__": "tlx_bench_shapes", "FocusSuite": shared["FocusSuite"], "FocusRegistry": shared["FocusRegistry"]
        }
        for module in shape_modules:
            shapes = ast.parse(self.module_path(module).read_text())
            shapes.body = [node for node in shapes.body if not (isinstance(node, ast.ImportFrom) and node.level)]
            exec(compile(shapes, str(self.module_path(module)), "exec"), namespace)
        namespace["SHAPE_SUITES"] = namespace["FOCUS"]
        exec(compile(ast.Module(body=[shape_filter], type_ignores=[]), "<bench shapes>", "exec"), namespace)
        selected = set()
        for arch in arches:
            namespace["driver"] = SimpleNamespace(arch=lambda: arch)
            if namespace["shapes"]():
                selected.add(arch)
        return selected

    def benchmarks(self):
        harness = self.repo / "python/test/tlx_benchmark"
        out = {}
        for op, modules in bench_modules(harness).items():
            for module in modules:
                if module.startswith("bench_torchtlx_"):
                    continue
                tree = ast.parse((harness / (module + ".py")).read_text())
                shape_modules = [
                    node.module
                    for node in tree.body
                    if isinstance(node, ast.ImportFrom) and any(alias.asname == "SHAPE_SUITES" for alias in node.names)
                ]
                arches = set().union(*(self.focus_arches(m) for m in shape_modules))
                if sum(not name.startswith("bench_torchtlx_") for name in modules) > 1:
                    arches = self.frontend_arches(tree, shape_modules, arches)
                covered = {op}
                if module == "bench_norm":
                    covered.update(spec.op
                                   for spec in self.specs
                                   if spec.impl.startswith("triton.tlx.ops.kernels.norm.") and spec.op != "rmsnorm_cat")
                instruments = {"third_party/tlx/ops/kernels/_shape_suites.py"}
                for shape_module in shape_modules:
                    instruments.update(self.dependencies(shape_module))
                out[module] = {"op": op, "ops": covered, "arches": arches, "instruments": sorted(instruments)}
        return out

    def plan(self, paths):
        affected = self.affected(paths)
        benches = self.benchmarks()
        plans = []
        for platform, arch in PLATFORMS.items():
            modules = {
                name: sorted(affected[platform] & bench["ops"])
                for name, bench in benches.items()
                if arch in bench["arches"] and any(spec.op == bench["op"] and spec.arch == arch
                                                   for spec in self.specs) and affected[platform] & bench["ops"]
            }
            if modules:
                plans.append({
                    "platform": platform,
                    "ops": sorted(affected[platform]),
                    "modules": modules,
                    "bench_ops": {name: benches[name]["op"]
                                  for name in modules},
                    "instruments": {name: benches[name]["instruments"]
                                    for name in modules},
                })
        return plans


def changed_paths(repo, before, after):
    return subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--name-only", "--no-renames", "-z", before, after, "--"
         ], ).decode().rstrip("\0").split("\0")


def select(repo, event):
    if event["ref"] != "refs/heads/main":
        return []
    paths = changed_paths(repo, event["before"], event["after"])
    plans = Catalog(repo).plan(paths)
    print(json.dumps({"changed": paths, "benchmarks": plans}, indent=2))
    return plans


def run_benchmarks(plan, output):
    import pytest

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    class Results:

        def pytest_runtest_setup(self, item):
            module = item.callspec.params["module_name"]
            item.config.option.json = str(output / (module + ".json"))

    result = pytest.main([
        *("python/test/tlx_benchmark/test_ops_perf.py::test_op_perf[" + module + "]" for module in plan["modules"]),
        "--import-mode=importlib",
        "-v",
        "-s",
        "--tb=short",
        "--timeout=2700",
        "--junitxml=" + str(output / "bench.xml"),
    ], plugins=[Results()])
    for module in plan["modules"]:
        path = output / (module + ".json")
        if not path.is_file() or not json.loads(path.read_text()).get("results"):
            print(f"Selected benchmark did not write measurement evidence: {module}")
            result = result or 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("select", "run"))
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.action == "select":
        if os.environ["GITHUB_EVENT_NAME"] != "push":
            plans = []
        else:
            event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
            plans = select(args.repo, event)
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            output.write("matrix=" + json.dumps(plans, separators=(",", ":")) + "\n")
    else:
        if args.out is None:
            parser.error("--out is required for run")
        raise SystemExit(run_benchmarks(json.loads(os.environ["BENCH_PLAN"]), args.out))


if __name__ == "__main__":
    main()
