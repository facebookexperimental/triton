#!/usr/bin/env python3
"""Parse LIT failures and positive recovery evidence for CI issue reporting."""
import argparse
import json
import os
import re

from classify_failure import classify
from test_results import encode_recovery

LIT_LINE = re.compile(r"^(?:FAIL|UNRESOLVED|TIMEOUT|XPASS):\s+(.*?)\s*(?:\(\d+ of \d+\))?\s*$")
RESULT_LINE = re.compile(r"^(PASS|FAIL|UNRESOLVED|TIMEOUT|XPASS|XFAIL|UNSUPPORTED):\s+(.*?)\s*(?:\(\d+ of \d+\))?\s*$")


def parse_log(path):
    ids = []
    seen = set()
    if not os.path.exists(path):
        return ids
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = LIT_LINE.match(line.strip())
            if not m:
                continue
            ident = m.group(1).strip()
            if ident and ident not in seen:
                seen.add(ident)
                ids.append(ident)
    return ids


def recovery_results(path, completed, failed, bucket):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            log = f.read()
    except OSError:
        return encode_recovery()
    results = {}
    for line in log.splitlines():
        m = RESULT_LINE.match(line.strip())
        if m:
            results[m[2]] = m[1]
    total = re.search(r"-- Testing: (\d+) tests", log)
    complete = bool(completed and total and int(total[1]) > 0 and len(results) == int(total[1])
                    and "Testing Time:" in log)
    passed = [ident for ident, status in results.items() if status == "PASS"]
    return encode_recovery(passed, [bucket] if complete and not failed and passed else [], complete)


def lit_repro(ident, fallback):
    """Best-effort local reproduce command for a LIT identity."""
    if fallback or "::" not in ident:
        return ""
    rel = ident.split("::", 1)[-1].strip()
    return f"lit -v test/{rel}" if rel else ""


def build_items(ids, workflow, job, fallback):
    items = []
    for ident in ids:
        summary = f"LIT test failed: {ident}"
        is_external, reason = classify(summary)
        items.append({
            "normalized_failure_id": ident,
            "raw_failure_ids": ident,
            "issue_title": f"[nightly] {workflow} / {job} / {ident}",
            "job_name": job,
            "summary": summary,
            "repro": lit_repro(ident, fallback),
            "fallback": fallback,
            "external_dep": is_external,
            "external_dep_reason": reason,
        })
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--bucket", default="compiler-lit")
    parser.add_argument("--failed", action="store_true")
    parser.add_argument("--completed", action="store_true")
    parser.add_argument("--output-name", default="failures")
    args = parser.parse_args()
    ids = parse_log(args.log)
    if not ids and not args.failed:
        items = []
    elif not ids:
        items = build_items([args.bucket], args.workflow, args.job, fallback=True)
    else:
        items = build_items(ids, args.workflow, args.job, fallback=False)
    payload = json.dumps(items)
    recovery = recovery_results(args.log, args.completed, args.failed, args.bucket)
    out_path = os.environ.get("GITHUB_OUTPUT")
    if out_path:
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(f"{args.output_name}={payload}\nrecovery={recovery}\n")
    print(payload)


if __name__ == "__main__":
    main()
