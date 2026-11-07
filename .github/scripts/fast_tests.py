"""Select fixed fast suites and require evidence that their tests ran."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from gpu_test_selection import changed_paths, select_platforms, test_plan, PATH_FILTERS, SUITES as FULL_SUITES
from run_selected_tests import command

ROOT = Path(__file__).resolve().parents[2]
SUITES = ROOT / ".github/fast-test-suites.json"
INTERFACE_PATHS = (".github/fast-test-suites.json", ".github/scripts/fast_tests.py",
                   ".github/workflows/fast-gpu-tests.yml", ".github/workflows/_fast-gpu-tests.yml",
                   ".github/actions/setup-test-env/action.yml")


def select(paths, filters):
    platforms = (dict.fromkeys(filters, True) if paths is None or any(path in INTERFACE_PATHS for path in paths) else
                 select_platforms(paths, filters))
    return platforms | {"cpu": any(platforms.values()), "build": any(platforms.values())}


def fast_plan(paths, repo, filters, suites, narrow):
    selected = select(paths, filters)
    plans = {platform: suites[platform] if selected[platform] else [] for platform in suites}
    if not narrow or not selected["build"] or paths is None or any(path in INTERFACE_PATHS for path in paths):
        return selected, plans
    full_suites = json.loads(FULL_SUITES.read_text())
    required = {platform: set() for platform in filters}
    for changed in paths:
        affected = select([changed], filters)
        if not affected["build"]:
            continue
        mapped = test_plan([changed], repo, filters, full_suites, narrow=True, offload=False)
        for platform, steps in mapped["gpu"].items():
            if not affected[platform]:
                continue
            targets = []
            if mapped["kernel"]["kernel_only"]:
                for target in suites[platform]:
                    path, name = target.split("::", 1)
                    if any(path in step["files"] and (not step["k"] or name in step["k"].split(" or "))
                           for step in steps.values()):
                        targets.append(target)
            required[platform].update(targets or suites[platform])
    for platform in filters:
        plans[platform] = [target for target in suites[platform] if target in required[platform]]
    return selected, plans


def verify_evidence(targets, junit):
    cases = list(ET.parse(junit).getroot().iter("testcase"))
    if not cases:
        raise ValueError("Fast correctness produced no test evidence")
    for case in cases:
        if any(case.find(tag) is not None for tag in ("skipped", "failure", "error")):
            raise ValueError(f"Fast correctness requires a passing result: {case.get('name')}")
    for target in targets:
        path, test = target.split("::", 1)
        classname = path.removesuffix(".py").replace("/", ".")
        if not any(
                case.get("classname") == classname and
            (case.get("name") == test or case.get("name", "").startswith(test + "[")) for case in cases):
            raise ValueError(f"Fast correctness did not execute required target: {target}")
    return len(cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("select", "run"))
    parser.add_argument("--platform", choices=("h100", "b200", "mi350", "cpu"))
    parser.add_argument("--junit")
    args = parser.parse_args()
    if args.action == "select":
        try:
            event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
            paths = changed_paths(os.environ["GITHUB_EVENT_NAME"], event)
        except (KeyError, ValueError, OSError, subprocess.CalledProcessError) as exc:
            print(f"::warning::Cannot select affected platforms; retaining all fast suites: {exc}")
            paths = None
        selected, plans = fast_plan(paths, ROOT, json.loads(PATH_FILTERS.read_text()), json.loads(SUITES.read_text()),
                                    narrow=os.environ["GITHUB_EVENT_NAME"] == "pull_request")
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            for key, value in selected.items():
                output.write(f"{key}={str(value).lower()}\n")
            for platform, targets in plans.items():
                output.write(f"{platform}_plan={json.dumps(targets, separators=(',', ':'))}\n")
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a") as output:
                platforms = ", ".join(key for key in ("h100", "b200", "mi350", "cpu") if selected[key])
                output.write("## Fast correctness selection\n\n" +
                             (f"Selected platforms: {platforms}. Required targets appear below.\n" if platforms else
                              "No fast GPU/compiler tests required for these paths. LIT is a separate signal.\n"))
                for platform, targets in plans.items():
                    if targets:
                        output.write(f"\n### {platform}\n\n" + "".join(f"- `{target}`\n" for target in targets))
        print(json.dumps({"selected": selected, "plans": plans}, indent=2))
        return
    if not args.platform or not args.junit:
        parser.error("run requires --platform and --junit")
    suite = json.loads(SUITES.read_text())[args.platform]
    targets = json.loads(os.environ["FAST_TARGETS"]) if "FAST_TARGETS" in os.environ else suite
    if not set(targets) <= set(suite):
        raise ValueError("The plan contains targets outside the declared fast suite")
    if not targets:
        raise ValueError("The fast suite is empty")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(ROOT / "python/test/unit/tlx_ops"), env.get("PYTHONPATH", "")))
    subprocess.run(command({"files": targets, "k": ""}, 1, args.junit), cwd=ROOT, env=env, check=True)
    count = verify_evidence(targets, args.junit)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as output:
            output.write(f"## {args.platform} fast correctness\n\n{count} passing test cases. "
                         "Every required target has passing evidence; no skipped cases.\n")
    print(f"Verified {count} passing cases for {len(targets)} required targets")


if __name__ == "__main__":
    main()
