"""Execute an already selected suite without recomputing its dependencies."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import re
import xml.etree.ElementTree as ET


def command(plan, workers, junit):
    files = plan["files"]
    if not files:
        raise ValueError("An allocated test step has no selected files")
    args = [
        sys.executable, "-m", "pytest", *files, "--import-mode=importlib", "-v", "-s", "--tb=short", "--durations=25",
        f"--junitxml={junit}"
    ]
    if plan["k"]:
        args.extend(["-k", plan["k"]])
    if workers > 1:
        args.extend(["-n", str(workers), "--dist", "worksteal"])
    return args


def check_cpu_skips(plan, junit):
    if "allowed_skips" not in plan:
        return
    for skipped in ET.parse(junit).getroot().iter("skipped"):
        reason = skipped.get("message", "").removeprefix("Skipped: ")
        if not any(re.fullmatch(pattern, reason) for pattern in plan["allowed_skips"]):
            raise ValueError(f"Unexpected skip in a CPU-only suite: {reason}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--junit", required=True)
    args = parser.parse_args()
    plan = json.loads(os.environ["TEST_SUITE"])
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(repo), str(repo / "python/test/unit/tlx_ops"), env.get("PYTHONPATH", "")])
    subprocess.run(command(plan, args.workers, args.junit), cwd=repo, env=env, check=True)
    check_cpu_skips(plan, args.junit)


if __name__ == "__main__":
    main()
