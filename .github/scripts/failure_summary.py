#!/usr/bin/env python3
"""Show a test job's parsed failures on its run page.

Reads the JSON failure list that parse_junit_failures.py or parse_lit_failures.py
prints, on stdin, and appends a markdown table to $GITHUB_STEP_SUMMARY, shown
under the job on the run's summary page. Prints nothing: the job log is what
internal CI signal cards read for their highlight. Writes nothing when the list
is empty.
"""
import json
import os
import sys


def cell(text):
    return (text or "").replace("|", "\\|").replace("\n", " ").strip()


def render(job, items):
    failing = [i for i in items if not i["normalized_failure_id"].startswith("slow-step:")]
    slow = [i for i in items if i["normalized_failure_id"].startswith("slow-step:")]
    lines = []
    if failing:
        lines += [
            f"### {job}: {len(failing)} failure{'s' if len(failing) != 1 else ''}", "",
            "| Test | Error | External dependency |", "|---|---|---|"
        ]
        for i in failing:
            ext = cell(i.get("external_dep_reason")) or "yes" if i.get("external_dep") else ""
            lines.append(f"| `{cell(i['normalized_failure_id'])}` | {cell(i['summary'])} | {ext} |")
        repros = [i["repro"] for i in failing if i.get("repro")]
        if repros:
            lines += ["", "Reproduce:", "", "```", *repros, "```"]
    if slow:
        lines += ["", f"### {job}: close to its time budget", ""]
        lines += [f"- {cell(i['summary'])}" for i in slow]
    return "\n".join(lines) + "\n" if lines else ""


def main():
    items = json.loads(sys.stdin.read().strip() or "[]")
    if not items:
        return
    job = os.environ["JOB"]
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(render(job, items))


if __name__ == "__main__":
    main()
