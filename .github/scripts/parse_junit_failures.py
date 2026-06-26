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

# Matches a single trailing bracketed parametrization suffix: test_x[a-b] -> test_x
PARAM_SUFFIX = re.compile(r"\[.*\]$")


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
        items.append({
            "normalized_failure_id": norm,
            "raw_failure_ids": "\n".join(info["raw"]),
            "issue_title": f"[nightly] {workflow} / {job} / {norm}",
            "job_name": job,
            "summary": info["summary"],
            "repro": pytest_repro(norm),
            # Real per-test signal: safe for reconcile to close on recovery.
            "fallback": False,
        })
    return items


def build_missing_junit_item(missing_paths, workflow, job):
    """Build a stable job-level item for pytest failures without JUnit XML."""
    norm = "pytest-junit-missing"
    missing = "\n".join(missing_paths)
    return {
        "normalized_failure_id": norm,
        "raw_failure_ids": missing,
        "issue_title": f"[nightly] {workflow} / {job} / {norm}",
        "job_name": job,
        "summary": f"Pytest failed before producing JUnit XML: {', '.join(missing_paths)}",
        "repro": "",
        # Job-level fallback: no per-test signal -> reconcile must not close.
        "fallback": True,
    }


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
    args = parser.parse_args()

    failures, missing_paths = collect_failures(args.junit)
    items = build_items(failures, args.workflow, args.job)
    if not items and missing_paths:
        items.append(build_missing_junit_item(missing_paths, args.workflow, args.job))
    if not items and args.failed:
        items.append(build_bucket_item(args.bucket, args.workflow, args.job))
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
