#!/usr/bin/env python3
"""Parse LLVM LIT output and emit failing-test identities for nightly CI.

LIT prints lines such as::

    FAIL: TRITON :: Conversion/foo.mlir (12 of 345)
    UNRESOLVED: TRITON :: Analysis/bar.mlir (3 of 345)

We extract the finest stable identity available (``TRITON :: <test path>``) for
each failing/unresolved/timed-out test. If none can be parsed but the run is
known to have failed, we fall back to a single stable logical bucket so a
failure is never silently dropped.

Output is written to ``$GITHUB_OUTPUT`` as ``failures=<json>`` (JSON array of
the same shape produced by ``parse_junit_failures.py``).
"""

import argparse
import json
import os
import re

# e.g. "FAIL: TRITON :: Conversion/foo.mlir (12 of 345)"
LIT_LINE = re.compile(
    r"^(?:FAIL|UNRESOLVED|TIMEOUT|XPASS):\s+(.*?)\s*(?:\(\d+ of \d+\))?\s*$"
)


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


def build_items(ids, workflow, job, bucket):
    if not ids:
        # Nothing parseable but the caller told us the run failed -> bucket.
        ids = [bucket]
    items = []
    for ident in ids:
        items.append(
            {
                "normalized_failure_id": ident,
                "raw_failure_ids": ident,
                "issue_title": f"[nightly] {workflow} / {job} / {ident}",
                "job_name": job,
                "summary": f"LIT test failed: {ident}",
            }
        )
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="LIT output log file")
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--bucket", default="compiler-lit", help="Fallback identity")
    parser.add_argument(
        "--failed",
        action="store_true",
        help="The LIT step failed; emit bucket if nothing parsed.",
    )
    parser.add_argument("--output-name", default="failures")
    args = parser.parse_args()

    ids = parse_log(args.log)
    if not ids and not args.failed:
        # Step passed and nothing parsed -> no issues.
        items = []
    else:
        items = build_items(ids, args.workflow, args.job, args.bucket)

    payload = json.dumps(items)
    out_path = os.environ.get("GITHUB_OUTPUT")
    if out_path:
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(f"{args.output_name}={payload}\n")
    print(payload)


if __name__ == "__main__":
    main()
