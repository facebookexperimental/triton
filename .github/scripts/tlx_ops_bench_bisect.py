#!/usr/bin/env python3
"""Bisect one nightly TLX throughput regression by building and measuring main commits on this GPU.

The worker runs from a tooling checkout outside --source. It checks out commits in --source, builds each into a
separate build venv, installs the wheel into the current (test) interpreter, and measures one case with the
pinned harness. Build, setup, missing-case, error and noisy results skip a commit; they never mark it bad.
"""
import argparse
import ast
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time

GOOD, BAD, SKIP = 0, 1, 125
MAX_CV = 0.03
VALID_RUNS = 2
MAX_ATTEMPTS = 4
MAX_JOBS = "8"
SHA = re.compile(r"^[0-9a-f]{40}$")
FIRST_BAD = re.compile(r"^([0-9a-f]{40}) is the first bad commit", re.M)
SELF = Path(__file__).resolve()


def bench_modules(harness):
    """Map each OP constant in the harness's bench_*.py files to the modules that declare it."""
    ops = {}
    for path in sorted(Path(harness).glob("bench_*.py")):
        for node in ast.parse(path.read_text()).body:
            if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "OP" for t in node.targets)
                    and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                ops.setdefault(node.value.value, []).append(path.stem)
    return ops


def valid_run(run):
    value, cv = run.get("tflops"), run.get("cv")
    return (run.get("status") in ("ok", "pip", "noisy") and isinstance(value, (int, float)) and math.isfinite(value)
            and value > 0 and isinstance(cv, (int, float)) and math.isfinite(cv) and 0 <= cv <= MAX_CV)


def summarize_runs(runs):
    """The harness's pip status (0.9x competitor floor) is ignored; only raw TLX TF/s and CV count."""
    if any(r.get("status") == "error" for r in runs):
        return {"status": "error", "runs": runs}
    values = [r["tflops"] for r in runs if valid_run(r)]
    if len(values) < VALID_RUNS:
        return {"status": "noisy", "runs": runs}
    return {"status": "ok", "values": values, "tflops": statistics.median(values), "runs": runs}


def classify(measurement, cutoff):
    if measurement.get("status") != "ok" or len(measurement.get("values", [])) < VALID_RUNS:
        return SKIP
    values = measurement["values"]
    if all(v >= cutoff for v in values):
        return GOOD
    if all(v < cutoff for v in values):
        return BAD
    return SKIP


def check_endpoints(good, bad, required_ratio):
    """Return (cutoff, None) when good/bad still reproduce the regression, else (None, reason)."""
    for name, m in (("good", good), ("bad", bad)):
        if m.get("status") != "ok":
            return None, f"{name} endpoint has no reliable measurement ({m.get('status')})"
    ratio = bad["tflops"] / good["tflops"]
    if ratio > required_ratio:
        return None, f"regression not reproduced: bad/good {ratio:.3f} > required {required_ratio:.3f}"
    cutoff = (good["tflops"] + bad["tflops"]) / 2
    if classify(good, cutoff) != GOOD or classify(bad, cutoff) != BAD:
        return None, "endpoint measurements overlap the midpoint cutoff"
    return cutoff, None


def first_bad(output, returncode):
    match = FIRST_BAD.search(output)
    return match.group(1) if returncode == 0 and match else None


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


def run(cmd, timeout=None, **kwargs):
    """Child output goes to stderr so git bisect's stdout stays parseable; a timeout kills the whole group."""
    if timeout is None:
        return subprocess.run(cmd, stdout=sys.stderr, check=True, **kwargs)
    proc = subprocess.Popen(cmd, stdout=sys.stderr, start_new_session=True, **kwargs)
    try:
        if proc.wait(timeout=max(1, timeout)):
            raise subprocess.CalledProcessError(proc.returncode, cmd)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        raise


def git(source, *args):
    return subprocess.run(["git", "-C", str(source), *args], capture_output=True, text=True, check=True).stdout


def is_ancestor(source, older, newer):
    return subprocess.run(["git", "-C", str(source), "merge-base", "--is-ancestor", older, newer]).returncode == 0


def build_install(step, timeout=None):
    source, work = Path(step["source"]), Path(step["work"])
    build, wheels, venv = work / "build", work / "wheels", work / "build-venv"
    restore(source)
    try:
        _build_install(step, source, build, wheels, venv, timeout)
    finally:
        restore(source)


def restore(source):
    """Keep the checkout pristine so the next checkout or bisect step cannot fail on build leftovers."""
    git(source, "reset", "-q", "--hard")
    git(source, "clean", "-ffdxq")


def _build_install(step, source, build, wheels, venv, timeout):
    for path in (build, wheels):
        shutil.rmtree(path, ignore_errors=True)
    if not (venv / "bin" / "python").exists():
        run(["uv", "venv", "--python", step["python"], str(venv)], timeout)
    env = {k: v for k, v in os.environ.items() if k not in ("TRITON_BUILD_WITH_CLANG_LLD", "TRITON_BUILD_WITH_CCACHE")}
    env.update(VIRTUAL_ENV=str(venv), PATH=f"{venv / 'bin'}{os.pathsep}{env.get('PATH', '')}",
               TRITON_BUILD_DIR=str(build), TRITON_WHEEL_NAME="fbtriton", MAX_JOBS=MAX_JOBS)
    py = str(venv / "bin" / "python")
    run(["uv", "pip", "install", "--python", py, "-r", "python/requirements.txt"], timeout, cwd=source, env=env)
    run(["uv", "build", "--wheel", "--no-build-isolation", "--python", py, "--out-dir",
         str(wheels)], timeout, cwd=source, env=env)
    built = sorted(wheels.glob("*.whl"))
    if len(built) != 1:
        raise RuntimeError(f"expected one wheel, found {len(built)}")
    run(["uv", "pip", "install", "--python", step["python"], "--force-reinstall", "--no-deps", str(built[0])], timeout)


def evaluate(step, timeout=None):
    """Build and measure the commit checked out in step["source"]; never raises for a broken commit."""
    try:
        build_install(step, timeout)
    except subprocess.TimeoutExpired:
        raise
    except Exception as exc:
        return {"status": "build_failed", "reason": f"{type(exc).__name__}: {exc}"}
    out = Path(step["work"]) / "measurement.json"
    out.unlink(missing_ok=True)
    try:
        run([step["python"], str(SELF), "--measure", step["path"], "--out", str(out)], timeout, cwd=step["work"])
        return json.loads(out.read_text())
    except subprocess.TimeoutExpired:
        raise
    except Exception as exc:
        return {"status": "error", "reason": f"measurement process failed: {type(exc).__name__}: {exc}"}


def measure_case(step):
    sys.path.insert(0, step["harness"])
    import triton
    if inside(triton.__file__, step["source"]):
        return {"status": "error", "reason": f"triton imported from the source tree: {triton.__file__}"}
    from _harness import driver
    from _harness.denoise import Governor, list_devices, select_device, stable

    # Every evaluation after the good endpoint measures on the GPU the good endpoint used.
    uuid = step.get("gpu_uuid")
    device = next((d for d in list_devices() if d.uuid == uuid), None) if uuid else select_device("auto")
    if device is None:
        return {"status": "error", "reason": f"GPU {uuid} not found"}
    driver.select(device)
    if not device.pin():
        return {"status": "error", "reason": f"gpu{device.index} has no UUID to pin"}
    matches = []
    for name in step["modules"]:
        bench = importlib.import_module(name)
        if bench.supported():
            matches += [(bench, c)
                        for c in bench.cases(False, None)
                        if c.key == step["key"] and c.direction == step["direction"]]
    if len(matches) != 1:
        return {"status": "missing_case", "reason": f"{len(matches)} cases match {step['key']}"}
    bench, case = matches[0]
    space = driver.resolve_space(bench, step.get("space"))
    runs = []
    with Governor(device), stable(driver.device_index()):
        for _ in range(MAX_ATTEMPTS):
            try:
                result = driver.run_case(bench, case, space=space, cold=False)
            except Exception as exc:
                runs.append({"status": "error", "notes": [f"{type(exc).__name__}: {exc}"]})
                break
            tlx = result.tlx
            runs.append({
                "status": result.status.value, "tflops": tlx.mean if tlx else None, "cv": tlx.cv if tlx else None,
                "notes": result.notes
            })
            print(f"{case.key}: {runs[-1]}", file=sys.stderr)
            if runs[-1]["status"] == "error" or sum(map(valid_run, runs)) >= VALID_RUNS:
                break
    return {
        **summarize_runs(runs), "device": device.name, "gpu_uuid": device.uuid, "space": space, "triton":
        triton.__file__
    }


def step_main(path):
    """One `git bisect run` step. Any failure exits 125 so it can never be counted as bad."""
    try:
        step = json.loads(Path(path).read_text())
        if not step.get("gpu_uuid"):
            raise ValueError("step has no GPU pinned by the good endpoint")
        sha = git(step["source"], "rev-parse", "HEAD").strip()
        measurement = evaluate(step)
        code = classify(measurement, step["cutoff"])
        with open(step["log"], "a") as log:
            log.write(json.dumps({"sha": sha, "code": code, **measurement}) + "\n")
        return code
    except Exception as exc:
        print(f"bisect step failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return SKIP


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    for field in ("op", "direction", "key", "good_sha", "bad_sha", "required_ratio", "platform"):
        if field not in spec:
            raise ValueError(f"spec is missing {field}")
    if not (SHA.match(spec["good_sha"]) and SHA.match(spec["bad_sha"])) or spec["good_sha"] == spec["bad_sha"]:
        raise ValueError("good_sha and bad_sha must be distinct full commit shas")
    if not 0 < float(spec["required_ratio"]) < 1:
        raise ValueError("required_ratio must be in (0, 1)")
    return spec


def bisect(args, result, deadline):
    spec = load_spec(args.spec)
    result.update({
        k: spec.get(k)
        for k in ("op", "direction", "key", "platform", "good_sha", "bad_sha", "required_ratio", "space")
    })
    source, harness = Path(args.source).resolve(), Path(args.harness).resolve()
    for name, path in (("worker", SELF), ("harness", harness)):
        if inside(path, source):
            return f"{name} must be outside the source checkout"
    good, bad = spec["good_sha"], spec["bad_sha"]
    if not is_ancestor(source, good, bad) or not is_ancestor(source, bad, "origin/main"):
        return "good_sha must be an ancestor of bad_sha, and bad_sha must be on origin/main"
    modules = bench_modules(harness).get(spec["op"])
    if not modules:
        return f"no bench module declares OP {spec['op']!r} in the harness"
    work = Path(tempfile.mkdtemp(prefix="tlx-bisect-", dir=os.environ.get("RUNNER_TEMP")))
    if inside(work, source):
        return "work directory must be outside the source checkout"
    step = {
        "source": str(source), "harness": str(harness), "work": str(work), "python": sys.executable, "modules": modules,
        "key": spec["key"], "direction": spec["direction"], "space": spec.get("space"), "log":
        str(work / "steps.jsonl"), "path": str(work / "step.json")
    }
    Path(step["path"]).write_text(json.dumps(step))
    endpoints = {}
    for name, sha in (("good", good), ("bad", bad)):
        run(["git", "-C", str(source), "checkout", "-q", "--detach", sha], deadline - time.time())
        endpoints[name] = evaluate(step, deadline - time.time())
        result["endpoints"] = endpoints
        if name == "good":
            if endpoints["good"].get("status") != "ok" or not endpoints["good"].get("gpu_uuid"):
                return f"good endpoint has no reliable measurement ({endpoints['good'].get('status')})"
            result["gpu_uuid"] = step["gpu_uuid"] = endpoints["good"]["gpu_uuid"]
            Path(step["path"]).write_text(json.dumps(step))
    cutoff, reason = check_endpoints(endpoints["good"], endpoints["bad"], float(spec["required_ratio"]))
    if reason:
        return reason
    result["cutoff"] = step["cutoff"] = cutoff
    Path(step["path"]).write_text(json.dumps(step))
    restore(source)
    started = git(source, "bisect", "start", bad, good, "--")
    try:
        if FIRST_BAD.search(started):  # good is bad's parent: nothing left to test
            output, returncode = started, 0
        else:
            cmd = ["git", "-C", str(source), "bisect", "run", sys.executable, str(SELF), "--step", step["path"]]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, start_new_session=True)
            try:
                output, _ = proc.communicate(timeout=max(1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.communicate()
                return "timed out during git bisect"
            returncode = proc.returncode
            print(output)
        result["bisect_log"] = git(source, "bisect", "log")
        sha = first_bad(output, returncode)
        if not sha:
            return "git bisect did not isolate one first bad commit (skipped or unmeasurable commits remain)"
        result.update(status="conclusive", first_bad_sha=sha, subject=git(source, "show", "-s", "--format=%s",
                                                                          sha).strip())
        return "first bad commit isolated between reproduced endpoints"
    finally:
        log = Path(step["log"])
        result["candidates"] = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        subprocess.run(["git", "-C", str(source), "bisect", "reset"], stdout=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source")
    parser.add_argument("--harness")
    parser.add_argument("--spec")
    parser.add_argument("--out")
    parser.add_argument("--timeout-minutes", type=float, default=540)
    parser.add_argument("--step", help=argparse.SUPPRESS)
    parser.add_argument("--measure", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.step:
        return step_main(args.step)
    if args.measure:
        try:
            measurement = measure_case(json.loads(Path(args.measure).read_text()))
        except Exception as exc:
            measurement = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
        Path(args.out).write_text(json.dumps(measurement))
        return 0
    if not all((args.source, args.harness, args.spec, args.out)):
        parser.error("--source, --harness, --spec and --out are required")
    result = {"status": "inconclusive"}
    try:
        result["reason"] = bisect(args, result, time.time() + args.timeout_minutes * 60)
    except subprocess.TimeoutExpired:
        result["reason"] = "timed out"
    except Exception as exc:
        result["reason"] = f"worker error: {type(exc).__name__}: {exc}"
    finally:
        Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"{result['status']}: {result['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
