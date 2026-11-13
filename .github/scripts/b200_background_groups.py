#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path

GROUP_PREFIX = "background-b200-"
GROUP_COUNT = 7
WORKERS = ("tlx", "torchtlx", "gluon")


def assign_groups(run_id, attempt, workers=WORKERS):
    slots = sorted(
        range(GROUP_COUNT),
        key=lambda slot: hashlib.sha256(f"{run_id}:{attempt}:{slot}".encode()).digest(),
    )
    return {worker: GROUP_PREFIX + str(slot) for worker, slot in zip(workers, slots)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID"))
    parser.add_argument("--attempt", type=int, default=int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")))
    parser.add_argument("--workers", default=",".join(WORKERS))
    parser.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    args = parser.parse_args()
    if not args.run_id:
        parser.error("--run-id or GITHUB_RUN_ID is required")
    workers = tuple(args.workers.split(","))
    if len(set(workers)) != len(workers) or not set(workers) <= set(WORKERS):
        parser.error("--workers must be a subset of tlx,torchtlx,gluon without duplicates")
    groups = assign_groups(args.run_id, args.attempt, workers)
    if args.github_output:
        with Path(args.github_output).open("a") as output:
            for worker, group in groups.items():
                output.write(f"{worker}_group={group}\n")
    print(json.dumps(groups, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
