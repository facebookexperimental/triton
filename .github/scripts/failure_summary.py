#!/usr/bin/env python3
"""Show a test job's parsed failures on its run page.

Reads the JSON failure list that parse_junit_failures.py or parse_lit_failures.py
wrote (FAILURES), then
  * appends a markdown table to $GITHUB_STEP_SUMMARY, shown under the job on the
    run's summary page, and
  * prints one ::error annotation per failure (::warning for slow steps), listed
    in the Annotations box at the top of that page.
Writes nothing when the list is empty.
"""
import json
import os

#: GitHub shows at most 10 error annotations per step.
MAX_ANNOTATIONS = 10


def cell(text):
    return (text or "").replace("|", "\\|").replace("\n", " ").strip()


def escape_data(text):
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def escape_property(text):
    return escape_data(text).replace(":", "%3A").replace(",", "%2C")


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


def annotations(job, items):
    out = []
    for i in items[:MAX_ANNOTATIONS]:
        level = "warning" if i["normalized_failure_id"].startswith("slow-step:") else "error"
        title = escape_property(f"{job}: {i['normalized_failure_id']}")
        out.append(f"::{level} title={title}::{escape_data(i['summary'] or i['normalized_failure_id'])}")
    return out


def main():
    items = json.loads(os.environ.get("FAILURES", "").strip() or "[]")
    if not items:
        return
    job = os.environ["JOB"]
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(render(job, items))
    print("\n".join(annotations(job, items)))


if __name__ == "__main__":
    main()
