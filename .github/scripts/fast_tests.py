"""Run the critical baseline plus representative tests for touched kernels."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

from gpu_test_selection import (
    changed_paths,
    classify,
    collect_imports,
    correctness_selection,
    iter_scan_files,
    manual_platforms,
    kernel_matchers_for,
    parse_catalog,
    select_platforms,
    test_file_affected,
    PATH_FILTERS,
    CORRECTNESS_REL,
)

ROOT = Path(__file__).resolve().parents[2]
SUITES = ROOT / ".github/fast-test-suites.json"
INTERFACE_PATHS = (".github/fast-test-suites.json", ".github/scripts/fast_tests.py",
                   ".github/workflows/fast-gpu-tests.yml", ".github/workflows/_fast-gpu-tests.yml",
                   ".github/actions/setup-test-env/action.yml", ".github/actions/install-torchao/action.yml")
MAX_CANDIDATES = 32


def select(paths, filters):
    platforms = (dict.fromkeys(filters, True) if paths is None or any(path in INTERFACE_PATHS for path in paths) else
                 select_platforms(paths, filters))
    return platforms | {"cpu": any(platforms.values()), "build": any(platforms.values())}


def fast_plan(paths, repo, filters, suites, narrow):
    selected = select(paths, filters)
    plans = {
        platform: {
            "baseline": list(suites[platform]) if selected[platform] else [], "touched": [], "unmapped": [], "limited":
            []
        }
        for platform in suites
    }
    if not narrow or not selected["build"] or paths is None:
        return selected, plans
    catalog, _ = parse_catalog(repo)
    kernels = [
        path for path in paths if classify(path) == "kernel" and not (path.startswith("third_party/tlx/ops/") and (
            not path.startswith("third_party/tlx/ops/kernels/") or Path(path).name == "_shapes.py"))
    ]
    matchers = {}
    for path in kernels:
        matcher = kernel_matchers_for([path], catalog)
        # Full CI keeps its incomplete-catalog fallback. Fast CI attributes only known op directories.
        if matcher.all_ops or matcher.all_tutorials or ("/ops/kernels/" in path and not matcher.op_names):
            continue
        matchers[path] = matcher
    mappings = {path: {} for path in kernels}
    patterns = {
        path:
        re.compile(r"\b(" + "|".join(re.escape(name)
                                     for name in sorted(matcher.op_names)) + r")\b") if matcher.op_names else None
        for path, matcher in matchers.items()
    }
    tests = {
        path
        for path in paths
        if classify(path) == "test" and path.endswith(".py") and (repo / path).is_file() and any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
            for node in ast.walk(ast.parse((repo / path).read_text())))
    }
    if matchers:
        for file in iter_scan_files(repo):
            source = (repo / file).read_text(encoding="utf-8")
            tree = ast.parse(source)
            if not any(
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
                    for node in ast.walk(tree)):
                continue
            imports = collect_imports(source)
            for path, matcher in matchers.items():
                if test_file_affected(source, file, matcher, patterns[path], imports=imports):
                    if file == CORRECTNESS_REL:
                        mode, expression = correctness_selection(repo, matcher, tests)
                        if mode != "skip":
                            mappings[path][file] = expression
                    else:
                        mappings[path][file] = ""
    covered_kernels = {
        path:
        any(
            select_platforms([path], filters)[platform] and selected[platform] and any(
                select_platforms([file], filters)[platform]
                for file in mapping)
            for platform in filters)
        for path, mapping in mappings.items()
    }
    for platform in filters:
        if not selected[platform]:
            continue
        touched = {}
        for path in paths:
            if not select_platforms([path], filters)[platform]:
                continue
            if path in kernels:
                mapping = mappings[path]
                compatible = {
                    file: expression
                    for file, expression in mapping.items()
                    if select_platforms([file], filters)[platform]
                }
                if not compatible and not covered_kernels[path]:
                    plans[platform]["unmapped"].append(path)
                elif not compatible:
                    plans[platform]["limited"].append({"path": path, "reason": "tests mapped on another platform"})
                for file, expression in compatible.items():
                    if file not in touched:
                        touched[file] = expression
                    elif touched[file] and expression:
                        touched[file] = " or ".join(sorted(set(touched[file].split(" or ") + expression.split(" or "))))
                    else:
                        touched[file] = ""
            elif path in tests:
                touched[path] = ""
            elif classify(path) == "test" and path.endswith(".py"):
                reason = "deleted test input" if not (repo / path).is_file() else "no pytest functions"
                plans[platform]["limited"].append({"path": path, "reason": reason})
            elif classify(path) in ("other", "kernel"):
                plans[platform]["limited"].append({"path": path, "reason": "shared input; baseline coverage only"})
        plans[platform]["touched"] = [{"file": file, "k": expression} for file, expression in sorted(touched.items())]
    return selected, plans


def target_matches(nodeid, target):
    return nodeid == target or nodeid.startswith(target + "[")


def case_matches(case, target):
    parts = target.split("::")
    classname = parts[0].removesuffix(".py").replace("/", ".")
    if len(parts) > 2:
        classname += "." + ".".join(parts[1:-1])
    return case.get("classname") == classname and target_matches(case.get("name", ""), parts[-1])


def verify_evidence(targets, junit, exclusions=()):
    cases = list(ET.parse(junit).getroot().iter("testcase"))
    if not cases or not targets:
        raise ValueError("Fast correctness produced no passing test evidence")
    for case in cases:
        if any(case.find(tag) is not None for tag in ("failure", "error")):
            raise ValueError(f"Fast correctness requires a passing result: {case.get('name')}")
        if case.find("skipped") is not None and not any(case_matches(case, item["nodeid"]) for item in exclusions):
            raise ValueError(f"Unreported skipped result: {case.get('name')}")
    for target in targets:
        if not any(case_matches(case, target) and case.find("skipped") is None for case in cases):
            raise ValueError(f"Fast correctness did not execute required target: {target}")
    return sum(case.find("skipped") is None for case in cases)


def declared_exclusion(item):
    for mark in item.iter_markers():
        if mark.name == "skip" or (mark.name in ("skipif", "xfail") and
                                   (any(condition is True
                                        for condition in mark.args) or mark.kwargs.get("condition") is True or
                                    (mark.name == "xfail" and not mark.args and "condition" not in mark.kwargs))):
            return str(mark.kwargs.get("reason", mark.name))
    return None


class FastCollection:

    def __init__(self, plan, repo):
        self.plan = plan
        self.baseline = []
        self.selected = []
        self.exclusions = []
        self.functions = {}
        self.files = {group["file"]: [] for group in plan["touched"]}
        self.reports = {}
        self.cases = {}
        for group in plan["touched"]:
            tree = ast.parse((repo / group["file"]).read_text())

            def inventory(nodes, parents=()):
                for node in nodes:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                        if not group["k"] or node.name in group["k"].split(" or "):
                            function = "::".join((group["file"], *parents, node.name))
                            self.functions[function] = {
                                "status": "missing", "passed": None, "attempts": 0, "reason": None
                            }
                            self.files[group["file"]].append(function)
                    elif isinstance(node, ast.ClassDef) and (node.name.startswith("Test") or any(
                            isinstance(base, ast.Attribute) and base.attr == "TestCase"
                            or isinstance(base, ast.Name) and base.id == "TestCase" for base in node.bases)):
                        inventory(node.body, (*parents, node.name))

            inventory(tree.body)

    def pytest_collection_modifyitems(self, session, config, items):
        missing = [
            target for target in self.plan["baseline"]
            if not any(target_matches(item.nodeid, target) for item in items)
        ]
        kept, deselected = [], []
        for item in items:
            function = item.nodeid.split("[", 1)[0]
            file = item.nodeid.split("::", 1)[0]
            baseline = any(target_matches(item.nodeid, target) for target in self.plan["baseline"])
            group = next((group for group in self.plan["touched"] if group["file"] == file and (
                not group["k"] or function.rsplit("::", 1)[-1] in group["k"].split(" or "))), None)
            if baseline or group:
                self.cases[item.nodeid] = "both" if baseline and group else "baseline" if baseline else "touched"
            if group:
                if function not in self.functions:
                    self.functions[function] = {"status": "missing", "passed": None, "attempts": 0, "reason": None}
                    self.files[file].append(function)
                reason = None if baseline else declared_exclusion(item)
                state = self.functions[function]
                if reason is None:
                    state["status"] = "pending"
                elif state["status"] == "missing":
                    state.update(status="excluded", reason=reason)
            else:
                reason = None
            if baseline or (group and reason is None):
                kept.append(item)
                if baseline:
                    self.baseline.append(item.nodeid)
            else:
                deselected.append(item)
                if group:
                    self.exclusions.append({"nodeid": item.nodeid, "phase": "declared", "reason": reason})
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept
        if missing:
            raise ValueError(f"Baseline targets were not collected: {missing}")

    def pytest_runtest_logreport(self, report):
        self.reports.setdefault(report.nodeid, []).append(report)

    def pytest_runtestloop(self, session):
        if session.testsfailed or session.config.option.collectonly:
            return None
        for item in session.items:
            function = item.nodeid.split("[", 1)[0]
            state = self.functions.get(function)
            baseline = item.nodeid in self.baseline
            if not baseline and state:
                if state["status"] in ("passed", "failed"):
                    continue
                if state["attempts"] >= MAX_CANDIDATES:
                    state["status"] = "limit_reached"
                    continue
                state["attempts"] += 1
            session.config.hook.pytest_runtest_protocol(item=item, nextitem=None)
            reports = self.reports.get(item.nodeid, [])
            failed = any(report.failed for report in reports)
            excluded = next((report for report in reports if report.skipped or hasattr(report, "wasxfail")), None)
            passed = any(report.when == "call" and report.passed
                         for report in reports) and not failed and excluded is None
            if failed:
                if state:
                    state["status"] = "failed"
            elif passed:
                self.selected.append(item.nodeid)
                if state:
                    state.update(status="passed", passed=item.nodeid)
            elif excluded is not None and not baseline:
                reason = str(excluded.longrepr)
                self.exclusions.append({"nodeid": item.nodeid, "phase": "runtime", "reason": reason})
                if state:
                    state["reason"] = reason
            if baseline and not passed and state:
                state["status"] = "failed"
        for state in self.functions.values():
            if state["status"] == "pending" and state["reason"]:
                state["status"] = "runtime_unavailable"
        return True

    def manifest(self):
        return {
            "platform": self.plan.get("platform", "local"), "baseline": self.baseline, "selected": self.selected,
            "files": self.files, "functions": self.functions, "cases": self.cases, "exclusions": self.exclusions,
            "unmapped": self.plan["unmapped"], "limited": self.plan.get("limited", [])
        }


def write_execution_summary(plan, manifest, junit, failure=None):
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return
    cases = list(ET.parse(junit).getroot().iter("testcase")) if Path(junit).is_file() else []
    rows = []
    matched = set()
    for nodeid, origin in manifest["cases"].items():
        case = next((case for case in cases if case_matches(case, nodeid)), None)
        exclusion = next((item for item in manifest["exclusions"] if item["nodeid"] == nodeid), None)
        if case is not None:
            matched.add(id(case))
            issue = next((case.find(tag) for tag in ("failure", "error", "skipped") if case.find(tag) is not None),
                         None)
            status = "pass" if issue is None else "skipped" if issue.tag == "skipped" else "fail"
            reason = "" if issue is None else issue.get("message") or issue.text or issue.tag
            duration = case.get("time", "unknown")
        else:
            status = "excluded" if exclusion else "not executed"
            reason = exclusion[
                "reason"] if exclusion else "not selected after a representative, failed attempt or attempt limit"
            duration = "-"
        rows.append((nodeid, origin, status, duration, reason))
    for target in plan["baseline"]:
        if not any(target_matches(nodeid, target) for nodeid in manifest["cases"]):
            rows.append((target, "baseline", "not executed", "-", "target was not collected"))
    for function, state in manifest["functions"].items():
        if not any(target_matches(nodeid, function) for nodeid in manifest["cases"]):
            rows.append((function, "touched", "not executed", "-", state["reason"] or "function was not collected"))
    for case in cases:
        if id(case) not in matched:
            issue = next((case.find(tag) for tag in ("failure", "error", "skipped") if case.find(tag) is not None),
                         None)
            rows.append((f"{case.get('classname', '')}::{case.get('name', '')}", "collection",
                         "fail" if issue is not None else "unmatched", case.get("time", "unknown"),
                         "" if issue is None else issue.get("message") or issue.text or issue.tag))

    def cell(text):
        return str(text).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    with open(summary, "a") as output:
        output.write(f"## {manifest['platform']} actual fast correctness results\n\n")
        if failure:
            output.write(f"Run failed: {cell(failure)}\n\n")
        output.write("| Exact test ID | Coverage | Actual result | JUnit seconds | Reason |\n|---|---|---|---|---|\n")
        for row in rows:
            output.write("| " + " | ".join(cell(value) for value in row) + " |\n")
        output.write(
            "\nNot-executed and excluded rows are not passes. Exact JUnit and selection JSON are in the test-results artifact.\n"
        )
        for function, state in manifest["functions"].items():
            if state["status"] != "passed":
                output.write(
                    f"\nFunction coverage: `{cell(function)}`, {cell(state['status'])}, {cell(state['reason'] or '')}.\n"
                )
        for path in manifest["unmapped"]:
            output.write(f"\nCoverage gap: `{cell(path)}` has no test mapping; baseline is not kernel validation.\n")


def run_plan(plan, repo, junit):
    import pytest
    from run_selected_tests import command
    collection = FastCollection(plan, repo)
    pytest.hookimpl(trylast=True)(FastCollection.pytest_collection_modifyitems)
    touched_files = [group["file"] for group in plan["touched"]]
    files = list(
        dict.fromkeys(
            [*(target for target in plan["baseline"] if target.split("::", 1)[0] not in touched_files),
             *touched_files]))
    if not files:
        raise ValueError("An allocated platform has no baseline or touched tests")
    old_cwd = Path.cwd()
    os.chdir(repo)
    try:
        result = pytest.main(command({"files": files, "k": ""}, 1, junit)[3:], plugins=[collection])
    finally:
        os.chdir(old_cwd)
    manifest = collection.manifest()
    failure = None
    try:
        if result != 0:
            raise ValueError(f"Fast pytest failed with exit code {result}")
        verify_evidence(collection.baseline, junit, collection.exclusions)
        verify_evidence(collection.selected, junit, collection.exclusions)
    except Exception as exc:
        failure = str(exc)
        raise
    finally:
        Path(junit).with_suffix(".selection.json").write_text(json.dumps(manifest, indent=2))
        write_execution_summary(plan, manifest, junit, failure)
    return manifest


def verify_manifests(manifests):
    """Coverage completeness is advisory; failures and claimed-pass integrity are not."""
    files, functions, passed, excluded, unaccounted = set(), set(), set(), set(), set()
    gaps, states = [], []
    for manifest in manifests:
        platform = manifest.get("platform", "local")
        for path in manifest["unmapped"]:
            gaps.append({
                "target":
                path, "details": [{
                    "platform": platform, "status": "unmapped", "reason":
                    "No test mapping; baseline is not touched-kernel coverage"
                }]
            })
        files.update(manifest["files"])
        functions.update(manifest["functions"])
        for function, state in manifest["functions"].items():
            states.append({
                "function": function, "platform": platform, "status": state["status"], "reason": state["reason"]
                or "No passing representative recorded"
            })
            if state["status"] == "failed":
                raise ValueError(f"Touched function did not complete: {function}: {state['status']}")
            if state["status"] == "passed":
                if state["passed"] not in manifest["selected"] or not target_matches(state["passed"], function):
                    raise ValueError(f"Touched function lacks matching executed evidence: {function}")
                passed.add(function)
            elif state["status"] == "excluded" and state["reason"] and any(
                    target_matches(item["nodeid"], function) for item in manifest["exclusions"]):
                excluded.add(function)
            else:
                unaccounted.add(function)
    missing = (functions - passed - excluded) | (unaccounted - passed)
    empty = {file for file in files if not any(function.startswith(file + "::") for function in passed)}
    for target in sorted(missing | empty):
        details = [
            state for state in states if state["function"] == target or state["function"].startswith(target + "::")
        ]
        gaps.append({"target": target, "details": details})
    warnings = []
    for gap in gaps:
        reasons = sorted({f"{detail['platform']}: {detail['status']}, {detail['reason']}" for detail in gap["details"]})
        message = f"{gap['target']}: {'; '.join(reasons) or 'No passing representative recorded'}"
        message = message.replace("\r", " ").replace("\n", " ")
        print(f"::warning::{message}")
        warnings.append(message)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if warnings and summary:
        with open(summary, "a") as output:
            output.write(
                "## Fast correctness coverage warnings\n\nCoverage gaps are advisory, not passing test evidence.\n\n" +
                "".join(f"- {message}\n" for message in warnings))
    return gaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("select", "run", "verify"))
    parser.add_argument("--platform", choices=("h100", "b200", "mi350", "cpu"))
    parser.add_argument("--junit")
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    if args.action == "verify":
        try:
            files = list(args.evidence.glob("**/*.selection.json"))
            if not files:
                raise ValueError("Selected workers produced no execution manifests")
            manifests = []
            for file in files:
                manifest = json.loads(file.read_text())
                junit = file.with_name(file.name.removesuffix(".selection.json") + ".xml")
                verify_evidence(manifest["baseline"], junit, manifest["exclusions"])
                verify_evidence(manifest["selected"], junit, manifest["exclusions"])
                manifests.append(manifest)
            expected = {
                key
                for key in ("h100", "b200", "mi350", "cpu")
                if json.loads(os.environ["NEEDS"])["select"]["outputs"][key] == "true"
            }
            if {manifest["platform"] for manifest in manifests} != expected:
                raise ValueError("Selected platforms have missing execution evidence")
            gaps = verify_manifests(manifests)
            print(json.dumps({"coverage_gaps": gaps}, sort_keys=True))
        except Exception as exc:
            summary = os.environ.get("GITHUB_STEP_SUMMARY")
            if summary:
                with open(summary, "a") as output:
                    output.write("## Aggregate fast correctness coverage failed\n\n" +
                                 str(exc).replace("\r", " ").replace("\n", " ") + "\n")
            raise
        return
    if args.action == "select":
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        paths = changed_paths(os.environ["GITHUB_EVENT_NAME"], event)
        selected, plans = fast_plan(paths, ROOT, json.loads(PATH_FILTERS.read_text()), json.loads(SUITES.read_text()),
                                    narrow=os.environ["GITHUB_EVENT_NAME"] == "pull_request")
        platforms = manual_platforms(os.environ["GITHUB_EVENT_NAME"], event)
        if platforms is not None:
            for platform in plans:
                if platform not in platforms:
                    selected[platform] = False
                    plans[platform] = {key: [] for key in plans[platform]}
        selected["b200"] = False
        plans["b200"] = {key: [] for key in plans["b200"]}
        selected["build"] = any(selected[platform] for platform in plans)
        disabled_note = "B200 temporarily disabled: runner offline; this run does not cover B200."
        print(f"::warning::{disabled_note}")
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            for key, value in selected.items():
                output.write(f"{key}={str(value).lower()}\n")
            for platform, plan in plans.items():
                output.write(f"{platform}_plan={json.dumps(plan, separators=(',', ':'))}\n")
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a") as output:
                output.write("## Baseline plus touched tests\n\n")
                output.write(disabled_note + "\n\n")
                if platforms is not None:
                    output.write(f"Manual platform list: {','.join(platforms)}. "
                                 "Other platforms are not covered by this run.\n\n")
                for platform, plan in plans.items():
                    if not selected[platform]:
                        continue
                    output.write(f"### {platform}\n\n{len(plan['baseline'])} baseline targets retained.\n\n")
                    for target in plan["baseline"]:
                        output.write(f"- Planned baseline: `{target}`\n")
                    for group in plan["touched"]:
                        output.write(f"- Touched tests: `{group['file']}` {group['k']}\n")
                    for path in plan["unmapped"]:
                        output.write(
                            f"- Coverage gap: `{path}` has no test mapping. The baseline does not validate it.\n")
                    for note in plan["limited"]:
                        output.write(f"- Limited coverage: `{note['path']}`, {note['reason']}.\n")
        print(json.dumps({"selected": selected, "plans": plans}, indent=2))
        return
    if not args.platform or not args.junit:
        parser.error("run requires --platform and --junit")
    baseline = json.loads(SUITES.read_text())[args.platform]
    plan = json.loads(os.environ["FAST_TARGETS"])
    if plan["baseline"] != baseline or not baseline:
        raise ValueError("Every selected platform must retain its complete baseline")
    repo_paths = (str(ROOT), str(ROOT / "python/test/unit/tlx_ops"))
    sys.path[:0] = repo_paths
    os.environ["PYTHONPATH"] = os.pathsep.join((*repo_paths, os.environ.get("PYTHONPATH", "")))
    plan["platform"] = args.platform
    manifest = run_plan(plan, ROOT, args.junit)
    print(f"Verified {len(manifest['selected'])} passing cases including the complete baseline")


if __name__ == "__main__":
    main()
