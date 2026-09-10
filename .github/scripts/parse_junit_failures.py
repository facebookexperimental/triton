#!/usr/bin/env python3
"""Parse pytest JUnit XML and emit a deduped list of failing test identities.

Used by nightly CI to file fine-grained GitHub issues per failing test.

Failing/errored testcases are collected from one or more JUnit XML files. If no
testcase failures can be parsed because expected XML was not produced, a stable
job-level fallback item is emitted. Each testcase's identity is normalized by
stripping the trailing parametrization suffix so that parameterized variants
collapse into a single issue, e.g.

    foo/test_tlx.py::test_bar[param_a]
    foo/test_tlx.py::test_bar[param_b]
        => foo/test_tlx.py::test_bar

A still-passing step whose runtime is closing on its ``timeout-minutes`` is also
reported, as a ``slow-step:<name>`` item, so it can be fixed before it starts
failing outright. Pass ``--budget <xml>=<minutes>`` to enable it.

The result is written to ``$GITHUB_OUTPUT`` as ``failures=<json>`` (a JSON array
of objects), defaulting to ``[]`` when there are no failures. Each object has:
    normalized_failure_id, raw_failure_ids, issue_title, job_name, summary
"""

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

from classify_failure import classify

# Matches a single trailing bracketed parametrization suffix: test_x[a-b] -> test_x
PARAM_SUFFIX = re.compile(r"\[.*\]$")

# Fraction of a step's timeout budget at which a still-passing suite is reported.
SLOW_THRESHOLD = 0.8


def normalize(node_id: str) -> str:
    return PARAM_SUFFIX.sub("", node_id)


def pytest_repro(norm: str) -> str:
    """Best-effort local reproduce command for a normalized pytest identity.

    JUnit ``classname`` is the dotted module path (e.g.
    ``python.test.unit.language.test_tlx_dot``), so convert it back to a file
    path and join the test function. Returns "" for non-pytest identities
    (job-level buckets without a ``::`` separator).
    """
    if "::" not in norm:
        return ""
    module, _, func = norm.partition("::")
    path = module.replace(".", "/") + ".py"
    return f"python -m pytest {path}::{func} -v"


def raw_id(classname: str, name: str) -> str:
    """Reconstruct a stable raw test id from JUnit classname + name."""
    classname = (classname or "").strip()
    name = (name or "").strip()
    if classname:
        return f"{classname}::{name}"
    return name


def collect_failures(paths):
    """Return parsed testcase failures and expected XML files that are missing."""
    failures = []
    missing_paths = []
    for path in paths:
        if not os.path.exists(path):
            # A missing XML usually means pytest failed before it could write
            # output, such as a timeout, import crash, or setup failure.
            missing_paths.append(path)
            continue
        try:
            tree = ET.parse(path)
        except ET.ParseError as exc:
            print(f"warning: could not parse {path}: {exc}", file=sys.stderr)
            continue
        root = tree.getroot()
        for tc in root.iter("testcase"):
            failure_nodes = tc.findall("failure") + tc.findall("error")
            if not failure_nodes:
                continue
            rid = raw_id(tc.get("classname", ""), tc.get("name", ""))
            msg = (failure_nodes[0].get("message") or "").strip()
            failures.append((rid, msg))
    return failures, missing_paths


def build_items(failures, workflow, job):
    """Group raw failures by normalized identity into reporter-ready items."""
    grouped = {}
    order = []
    for rid, msg in failures:
        norm = normalize(rid)
        if norm not in grouped:
            grouped[norm] = {"raw": [], "summary": ""}
            order.append(norm)
        if rid not in grouped[norm]["raw"]:
            grouped[norm]["raw"].append(rid)
        if not grouped[norm]["summary"] and msg:
            grouped[norm]["summary"] = msg.splitlines()[0][:300]

    items = []
    for norm in order:
        info = grouped[norm]
        is_external, reason = classify(info["summary"])
        items.append({
            "normalized_failure_id": norm,
            "raw_failure_ids": "\n".join(info["raw"]),
            "issue_title": f"[nightly] {workflow} / {job} / {norm}",
            "job_name": job,
            "summary": info["summary"],
            "repro": pytest_repro(norm),
            # Real per-test signal: safe for reconcile to close on recovery.
            "fallback": False,
            # External-dep failures skip bisection and are annotated instead.
            "external_dep": is_external,
            "external_dep_reason": reason,
        })
    return items


def build_missing_junit_item(missing_paths, workflow, job):
    """Build a stable job-level item for pytest failures without JUnit XML."""
    norm = "pytest-junit-missing"
    missing = "\n".join(missing_paths)
    summary = f"Pytest failed before producing JUnit XML: {', '.join(missing_paths)}"
    is_external, reason = classify(summary)
    return {
        "normalized_failure_id": norm,
        "raw_failure_ids": missing,
        "issue_title": f"[nightly] {workflow} / {job} / {norm}",
        "job_name": job,
        "summary": summary,
        "repro": "",
        # Job-level fallback: no per-test signal -> reconcile must not close.
        "fallback": True,
        "external_dep": is_external,
        "external_dep_reason": reason,
    }


def parse_budget(spec):
    """Parse a ``--budget`` pair, ``<junit-xml-path>=<timeout-minutes>``."""
    path, sep, minutes = spec.rpartition("=")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected <path>=<minutes>, got {spec!r}")
    try:
        return path, float(minutes)
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad minutes in {spec!r}") from None


def suite_seconds(path):
    """Total test-session seconds recorded in a JUnit XML, or None if unusable."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None
    suites = list(root.iter("testsuite"))
    if not suites:
        return None
    total = 0.0
    for suite in suites:
        try:
            total += float(suite.get("time") or 0.0)
        except ValueError:
            continue
    return total


def build_slow_items(budgets, threshold, workflow, job):
    """Report each still-passing step whose runtime is closing on its timeout."""
    items = []
    for path, budget_min in budgets:
        seconds = suite_seconds(path)
        if seconds is None or budget_min <= 0:
            continue
        used_min = seconds / 60.0
        frac = used_min / budget_min
        if frac < threshold:
            continue
        name = os.path.basename(path)
        norm = f"slow-step:{name[:-4] if name.endswith('.xml') else name}"
        items.append({
            "normalized_failure_id":
            norm,
            "raw_failure_ids":
            "",
            "issue_title":
            f"[nightly] {workflow} / {job} / {norm}",
            "job_name":
            job,
            "summary": (f"{name}: {used_min:.1f} min of a {budget_min:g} min budget ({frac:.0%}). This is "
                        f"test-session time only, excluding collection, so the step is slower still. "
                        f"Speed the suite up or raise timeout-minutes before it starts timing out."),
            # No repro: this is a whole suite, not one test. An empty repro also
            # tells report-nightly-failure.yml to skip deep bisection.
            "repro":
            "",
            # Real, per-step signal: safe for reconcile to close once it speeds up.
            "fallback":
            False,
            "external_dep":
            False,
            "external_dep_reason":
            "",
        })
    return items


def build_bucket_item(bucket, workflow, job):
    """Build a stable job-level fallback item when the job failed but no test failures can be parsed."""
    return {
        "normalized_failure_id": bucket,
        "raw_failure_ids": "",
        "issue_title": f"[nightly] {workflow} / {job} / {bucket}",
        "job_name": job,
        "summary": "Job failed but no test failures could be parsed (see run log).",
        "repro": "",
        # Job-level fallback: no per-test signal -> reconcile must not close.
        "fallback": True,
        # No parseable summary to classify; treat as a code failure (not external).
        "external_dep": False,
        "external_dep_reason": "",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", nargs="+", required=True, help="JUnit XML file(s)")
    parser.add_argument("--workflow", required=True, help="Workflow name (for title)")
    parser.add_argument("--job", required=True, help="Job name (for title)")
    parser.add_argument("--output-name", default="failures", help="GITHUB_OUTPUT key to write")
    parser.add_argument(
        "--failed",
        action="store_true",
        help=("Explicitly signal that the job/step failed; emit a fallback item "
              "if no test failures can be parsed from JUnit XML."),
    )
    parser.add_argument(
        "--bucket",
        default="job-failed-no-parseable-failures",
        help="Stable bucket id for the fallback item emitted when --failed is set.",
    )
    parser.add_argument(
        "--budget",
        action="append",
        default=[],
        metavar="XML=MINUTES",
        help=("Step timeout budget for a JUnit XML, e.g. /tmp/tlx-core.xml=30. "
              "Repeatable. Enables near-timeout reporting for that file."),
    )
    parser.add_argument(
        "--slow-threshold",
        type=float,
        default=SLOW_THRESHOLD,
        help="Fraction of the budget at which a passing-but-slow step is reported.",
    )
    args = parser.parse_args()

    failures, missing_paths = collect_failures(args.junit)
    items = build_items(failures, args.workflow, args.job)
    if not items and missing_paths:
        items.append(build_missing_junit_item(missing_paths, args.workflow, args.job))
    if not items and args.failed:
        items.append(build_bucket_item(args.bucket, args.workflow, args.job))
    # Appended after the fallback guards: a slow-step item must not suppress the
    # job-level fallback that a genuine failure needs.
    items += build_slow_items([parse_budget(b) for b in args.budget], args.slow_threshold, args.workflow, args.job)
    payload = json.dumps(items)

    # TODO(scuba): in a follow-up, also emit a metrics row per failure
    # ({workflow, job, normalized_id, status, sha, run_id}) to a Scuba table.
    # Feasible on the self-hosted GPU runners (scribe reachable); skip on the
    # GitHub-hosted ubuntu-latest LIT job, which has no Scuba access.
    out_path = os.environ.get("GITHUB_OUTPUT")
    if out_path:
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(f"{args.output_name}={payload}\n")
    # Always echo to stdout for log visibility / local debugging.
    print(payload)


if __name__ == "__main__":
    main()
