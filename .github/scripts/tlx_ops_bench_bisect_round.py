#!/usr/bin/env python3
"""Select and classify one main-history commit per throughput-bisect round."""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from tlx_ops_bench_bisect import BAD, FIRST_BAD, GOOD, SHA, SKIP, check_endpoints, classify

DEFAULT_MAX_ROUNDS = 16
STATE_FIELDS = ("specs", "index", "observations", "results", "round", "max_rounds", "verified", "log")
SPEC_FIELDS = ("op", "direction", "key", "platform", "good_sha", "bad_sha", "required_ratio")
ONLY_SKIPPED = re.compile(r"only 'skip'ped commits left to test")
REVISIONS_LEFT = re.compile(r"Bisecting: (\d+) revisions? left to test after this \(roughly (\d+) steps?\)")
GIT_ENV = {**os.environ, "LC_ALL": "C", "LANG": "C", "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1"}


class Invalid(Exception):
    """The current spec cannot be bisected as recorded; it finishes inconclusive."""


def git(cwd, *args, check=True):
    proc = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, env=GIT_ENV)
    if check and proc.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc


def require_commit(source, sha):
    if git(source, "cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode:
        raise Invalid(f"commit {sha} is not in the checkout")


def is_ancestor(source, older, newer):
    proc = git(source, "merge-base", "--is-ancestor", older, newer, check=False)
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"git merge-base --is-ancestor {older} {newer} failed: {proc.stderr.strip()}")
    return proc.returncode == 0


def validate_spec(source, spec):
    for field in SPEC_FIELDS:
        if field not in spec:
            raise Invalid(f"spec is missing {field}")
    good, bad = spec["good_sha"], spec["bad_sha"]
    if not (isinstance(good, str) and isinstance(bad, str) and SHA.fullmatch(good)
            and SHA.fullmatch(bad)) or good == bad:
        raise Invalid("good_sha and bad_sha must be distinct full commit shas")
    ratio = spec["required_ratio"]
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or not 0 < ratio < 1:
        raise Invalid("required_ratio must be in (0, 1)")
    require_commit(source, good)
    require_commit(source, bad)
    if git(source, "rev-parse", "--verify", "-q", "origin/main^{commit}", check=False).returncode:
        raise RuntimeError("the source checkout has no origin/main")
    if not is_ancestor(source, good, bad) or not is_ancestor(source, bad, "origin/main"):
        raise Invalid("good_sha must be an ancestor of bad_sha, and bad_sha must be on origin/main")


def narrow(source, spec, observations):
    """(newest known-bad boundary, known-good commits, skipped commits) after checking the history is consistent.

    git keeps a single bad commit, so the replay must start from the oldest one measured bad. A measured good
    commit that is not older than it means the throughput did not move monotonically along the history.
    """
    good, bad = spec["good_sha"], spec["bad_sha"]
    goods, bads, skips = [good], [bad], []
    for sha, code in sorted(observations.items()):
        if not (isinstance(sha, str) and SHA.fullmatch(sha)) or isinstance(code, bool) or code not in (GOOD, BAD, SKIP):
            raise Invalid(f"invalid observation {sha!r}: {code!r}")
        if sha in (good, bad):
            raise Invalid(f"endpoint {sha} was recorded as an observation")
        require_commit(source, sha)
        if not is_ancestor(source, sha, bad) or is_ancestor(source, sha, good):
            raise Invalid(f"observed commit {sha} is outside the good..bad range")
        {GOOD: goods, BAD: bads, SKIP: skips}[code].append(sha)
    boundary = [b for b in bads if all(is_ancestor(source, b, other) for other in bads)]
    if len(boundary) != 1:
        raise Invalid("commits measured bad do not lie on one line of history")
    for sha in goods:
        if is_ancestor(source, boundary[0], sha):
            raise Invalid(f"commit {sha} measured good is not older than {boundary[0]} measured bad: "
                          "throughput is not monotone along the history")
    return boundary[0], goods, skips


def replay(source, boundary, goods, skips):
    """What git bisect says after these observations, run in a shared no-checkout clone that is then deleted."""
    with tempfile.TemporaryDirectory(prefix="tlx-bisect-round-") as tmp:
        repo = Path(tmp) / "repo"
        subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout",
                        str(source), str(repo)], check=True, capture_output=True, env=GIT_ENV)
        step = git(repo, "bisect", "start", "--no-checkout", boundary, *goods, "--", check=False)
        if skips:
            step = git(repo, "bisect", "skip", *skips, check=False)
        output = step.stdout + step.stderr
        found = FIRST_BAD.search(output)
        if found:
            if step.returncode:
                raise RuntimeError(f"git bisect failed while naming a commit: {output.strip()}")
            if found[1] != boundary:
                raise RuntimeError(f"git named {found[1]} as the first bad commit but the bad boundary is {boundary}")
            return {"kind": "first_bad", "sha": boundary}
        if ONLY_SKIPPED.search(output):
            tail = output.split("could be any of:", 1)[-1]
            return {"kind": "skipped_only", "commits": re.findall(r"^[0-9a-f]{40}$", tail, re.M)}
        head = git(repo, "rev-parse", "--verify", "-q", "BISECT_HEAD", check=False)
        sha = head.stdout.strip()
        if head.returncode or not SHA.fullmatch(sha):
            raise RuntimeError(f"git bisect gave no candidate (exit {step.returncode}): {output.strip()}")
        left = REVISIONS_LEFT.search(output)
        return {
            "kind": "candidate", "sha": sha, "revisions_left": int(left[1]) if left else None, "steps_left":
            int(left[2]) if left else None
        }


def normalize_state(raw):
    if not isinstance(raw, dict) or not isinstance(raw.get("specs"), list) or not all(
            isinstance(s, dict) for s in raw["specs"]):
        raise ValueError("state needs a `specs` list of objects")
    state = {
        "specs": copy.deepcopy(raw["specs"]), "index": raw.get("index", 0), "observations":
        copy.deepcopy(raw.get("observations", {})), "results": copy.deepcopy(raw.get("results", [])), "round":
        raw.get("round", 0), "max_rounds": raw.get("max_rounds", DEFAULT_MAX_ROUNDS), "verified":
        raw.get("verified", False), "log": copy.deepcopy(raw.get("log", []))
    }
    ints = (state["index"], state["round"], state["max_rounds"])
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in ints):
        raise ValueError("index, round and max_rounds must be integers")
    if not 0 <= state["index"] <= len(state["specs"]) or state["round"] < 0 or not 1 <= state["max_rounds"] <= 64:
        raise ValueError("index, round or max_rounds is out of range")
    if not isinstance(state["observations"], dict) or not isinstance(state["results"], list) or not isinstance(
            state["log"], list) or not isinstance(state["verified"], bool):
        raise ValueError("observations, results, log and verified have the wrong types")
    return state


def result(state, spec, status, reason, **extra):
    return {
        "status": status, "reason": reason, "issue_number": spec.get("issue_number"), "key": spec.get("key"), "op":
        spec.get("op"), "direction": spec.get("direction"), "platform": spec.get("platform"), "good_sha":
        spec.get("good_sha"), "bad_sha": spec.get("bad_sha"), "required_ratio": spec.get("required_ratio"), "space":
        spec.get("space"), "observations": dict(sorted(state["observations"].items())), "rounds":
        copy.deepcopy(state["log"]), **extra
    }


def prepare(raw_state, source):
    state = normalize_state(raw_state)
    prepared = {
        **state, "done": False, "spec": None, "candidate_sha": None, "first_bad_sha": None, "first_bad_parents": [],
        "commits": [], "needs_measurement": False, "finish": None, "next_inputs": None, "revisions_left": None,
        "steps_left": None
    }
    if state["index"] >= len(state["specs"]):
        prepared["done"] = True
        return prepared
    spec = state["specs"][state["index"]]
    prepared["spec"] = spec
    try:
        validate_spec(source, spec)
        found = replay(source, *narrow(source, spec, state["observations"]))
    except Invalid as exc:
        prepared["finish"] = result(state, spec, "inconclusive", str(exc))
        return prepared
    if found["kind"] == "skipped_only":
        listed = ", ".join(found["commits"])
        prepared["finish"] = result(state, spec, "inconclusive",
                                    "Only skipped commits remain, so the first bad commit could be any of: " + listed,
                                    candidates=found["commits"])
        return prepared
    if found["kind"] == "candidate" and state["round"] >= state["max_rounds"]:
        prepared["finish"] = result(state, spec, "inconclusive",
                                    f"No first bad commit after {state['round']} rounds (max_rounds).")
        return prepared
    good, bad = spec["good_sha"], spec["bad_sha"]
    commits = [good, bad]
    if found["kind"] == "first_bad":
        prepared["first_bad_sha"] = found["sha"]
        prepared["first_bad_parents"] = git(source, "rev-list", "--parents", "-n", "1", found["sha"]).stdout.split()[1:]
    else:
        prepared["candidate_sha"] = found["sha"]
        prepared["revisions_left"], prepared["steps_left"] = found["revisions_left"], found["steps_left"]
        commits.append(found["sha"])
    prepared["commits"] = commits
    prepared["needs_measurement"] = True
    prepared["next_inputs"] = {
        **{k: spec.get(k)
           for k in ("issue_number", "op", "direction", "key", "platform", "space", "required_ratio")}, "good_sha":
        good, "bad_sha": bad, "candidate_sha": prepared["candidate_sha"]
    }
    return prepared


def settle(state, finished):
    """Record the current spec's result and start the next spec with a clean slate."""
    state["results"].append(finished)
    state.update(index=state["index"] + 1, observations={}, round=0, verified=False, log=[])
    return state


def unusable(measurement):
    return not isinstance(measurement, dict) or measurement.get("status") == "build_failed" or bool(
        measurement.get("build_failed"))


def advance(prepared, measurements):
    state = normalize_state(prepared)
    if state["index"] >= len(state["specs"]):
        return state
    spec = state["specs"][state["index"]]
    finished = prepared.get("finish")
    if finished:
        return settle(state, finished)
    measurements = measurements or {}
    good, bad, candidate = measurements.get("good"), measurements.get("bad"), measurements.get("candidate")
    for name, m in (("good", good), ("bad", bad)):
        if unusable(m):
            return settle(
                state,
                result(
                    state, spec, "inconclusive", f"The {name} endpoint has no measurement "
                    f"({m.get('status') if isinstance(m, dict) else 'missing'})."))
    if any(isinstance(m, dict) and m.get("status") == "ok" and not m.get("gpu_uuid") for m in (good, bad, candidate)):
        return settle(state, result(state, spec, "inconclusive", "A valid measurement has no GPU UUID."))
    gpus = {m.get("gpu_uuid") for m in (good, bad, candidate) if isinstance(m, dict) and m.get("gpu_uuid")}
    if len(gpus) > 1:
        return settle(
            state, result(state, spec, "inconclusive",
                          "The endpoints and candidate were not measured on the same GPU."))
    cutoff, reason = check_endpoints(good, bad, float(spec["required_ratio"]))
    evidence = {"endpoints": {"good": good, "bad": bad}, "cutoff": cutoff, "gpu_uuid": next(iter(gpus), None)}
    if reason:
        return settle(state, result(state, spec, "inconclusive", reason, endpoints={"good": good, "bad": bad}))
    state["verified"] = True
    entry = {
        "round": state["round"], "candidate": prepared.get("candidate_sha"), "code": None, "cutoff": cutoff,
        "good_tflops": good["tflops"], "bad_tflops": bad["tflops"], "gpu_uuid": evidence["gpu_uuid"]
    }
    if prepared.get("candidate_sha") is None:
        first_bad = prepared.get("first_bad_sha")
        if not first_bad:
            raise ValueError("prepared state has neither a candidate nor a first bad commit")
        state["log"].append(entry)
        return settle(
            state,
            result(state, spec, "conclusive", "First bad commit isolated between endpoints that "
                   "reproduced the regression.", first_bad_sha=first_bad,
                   first_bad_parents=prepared.get("first_bad_parents", []), verified=True, **evidence))
    sha = prepared["candidate_sha"]
    if sha in state["observations"]:
        raise ValueError(f"candidate {sha} was already measured")
    code = SKIP if unusable(candidate) else classify(candidate, cutoff)
    state["observations"][sha] = code
    entry.update(code=code, candidate_tflops=candidate.get("tflops") if isinstance(candidate, dict) else None,
                 candidate_status=candidate.get("status") if isinstance(candidate, dict) else "missing")
    state["log"].append(entry)
    state["round"] += 1
    return state


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--state", required=True)
    prep.add_argument("--source", required=True)
    prep.add_argument("--out", required=True)
    adv = sub.add_parser("advance")
    adv.add_argument("--state", required=True, help="the prepared state of the round that was measured")
    adv.add_argument("--measurements", help="omit only when the prepared state finishes without measuring")
    adv.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        state = json.loads(Path(args.state).read_text())
        if args.command == "prepare":
            prepared = prepare(state, args.source)
            dump(args.out, prepared)
            if prepared["done"]:
                print("done: every spec has a result")
            elif prepared["finish"]:
                print(f"finish without measuring: {prepared['finish']['reason']}")
            else:
                print(f"round {prepared['round']}: measure {' '.join(prepared['commits'])}")
        else:
            measurements = json.loads(Path(args.measurements).read_text()) if args.measurements else {}
            if prepared_needs_measurement(state) and not args.measurements:
                parser.error("--measurements is required for a round that measures commits")
            dump(args.out, advance(state, measurements))
    except (RuntimeError, ValueError, KeyError) as exc:
        raise SystemExit(f"error: {exc}")
    return 0


def prepared_needs_measurement(prepared):
    return bool(prepared.get("needs_measurement")) and not prepared.get("finish") and not prepared.get("done")


if __name__ == "__main__":
    raise SystemExit(main())
