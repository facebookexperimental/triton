#!/usr/bin/env python3
"""Measure one prebuilt wheel in a performance-bisect round."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import tlx_ops_bench_bisect as worker


def measure(prepared, role, source, harness, output):
    spec = prepared["spec"]
    step = {**spec, "source": str(source.resolve()), "harness": str(harness.resolve())}
    step["modules"] = worker.bench_modules(harness).get(spec["op"], [])
    if role != "good":
        good = json.loads((output / "good.json").read_text())
        if not good.get("gpu_uuid"):
            return {"status": "error", "reason": "good endpoint did not select a GPU"}
        step["gpu_uuid"] = good["gpu_uuid"]
    work = Path(tempfile.mkdtemp(prefix=f"perf-measure-{role}-", dir=os.environ.get("RUNNER_TEMP")))
    step_path = work / "step.json"
    step_path.write_text(json.dumps(step))
    destination = work / "measurement.json"
    env = os.environ.copy()
    env["TRITON_HOME"] = str(output.resolve() / "triton-home")
    for key in (*worker.GIT_REPO_ENV, "ACTIONS_RUNTIME_TOKEN", "ACTIONS_RUNTIME_URL", "ACTIONS_CACHE_URL",
                "ACTIONS_RESULTS_URL", "ACTIONS_ID_TOKEN_REQUEST_TOKEN", "ACTIONS_ID_TOKEN_REQUEST_URL", "GH_TOKEN",
                "GITHUB_TOKEN", "GITHUB_ENV", "GITHUB_PATH", "GITHUB_OUTPUT", "GITHUB_STATE", "GITHUB_STEP_SUMMARY"):
        env.pop(key, None)
    worker.run(
        [sys.executable, str(worker.SELF), "--measure",
         str(step_path), "--out",
         str(destination)], timeout=900, cwd=work, env=env)
    if not destination.exists():
        return {"status": "error", "reason": "measurement process failed"}
    return json.loads(destination.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--role", choices=("good", "bad", "candidate"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        result = measure(json.loads(args.prepared.read_text()), args.role, args.source, args.harness, args.output)
    except Exception as exc:
        result = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
    (args.output / f"{args.role}.json").write_text(json.dumps(result, indent=2))
    print(f"{args.role}: {result['status']}")


if __name__ == "__main__":
    main()
