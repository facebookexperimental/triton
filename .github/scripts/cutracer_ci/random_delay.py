#!/usr/bin/env python3
"""Run tests under CUTracer random delay and report the races it exposes.

Correct synchronization does not depend on timing, so a test that fails when
CUTracer delays its threads at random points suggests a race the sanitizers
cannot see. Each test target runs in its own processes, because a kernel fault
leaves a sticky CUDA error that fails every later test in the same process:

    plain    pytest without CUTracer
    delay    pytest under ``cutracer trace -a random_delay``
    confirm  each test that passed plain and failed under delay, alone under
             delay again

Every run records each test's result as it finishes (record_outcomes.py), so a
process that dies mid-run, which CUTracer can do while instrumenting a kernel,
loses only the test it was running. A CUDA error counts the same way, since it
leaves a sticky error that fails every later test in the process. That test is
reported as crashed and the delay run resumes with the tests after it.

A test recorded passing plain and failing both delay runs is a race the delay
exposes. With ``--issue-title-prefix``, each such test function becomes one
entry in ``failures=<json>`` on ``$GITHUB_OUTPUT``, after the sanitizer step's
entries, for report-nightly-failure.yml. Everything else -- a test that fails
plain, is skipped, crashed or never ran -- appears only in the job summary.
Nothing here closes these issues: one clean run under random delay does not
show a race is gone.

Exit status: 1 when a test function without an open issue (``--known-races``)
has a race, else 0. A race already on main, a crash, or a failure that did not
repeat must not fail a pull request that did not cause it.

Usage:
    random_delay.py --cutracer-so lib/cutracer.so --out /tmp/rd --logs /tmp/rd-logs \\
        python/test/unit/language/test_tlx_dot_sm100.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from findings import MAX_SUMMARY_CHARS, failure_entry  # noqa: E402
from parse_junit_failures import normalize  # noqa: E402

PYTEST = ["python", "-m", "pytest", "--import-mode=importlib", "-q", "-p", "no:cacheprovider", "-p", "record_outcomes"]
# A delay run that keeps dying is not worth more of the step's time.
MAX_RESUMES = 3
# Confirmation runs one process per test; keep a target's cost bounded.
MAX_CONFIRMS = 8
CUDA_ERROR = re.compile(r"CUDA error|cudaError|AcceleratorError|Triton Error \[CUDA\]")
# CUTracer prints a line per delay point; keep them out of the job log.
NOISE = re.compile(r"^(Delay injection|CUTracer: )")


@dataclass
class Run:
    """What one pytest process recorded before it ended."""
    outcomes: dict[str, tuple[str, str]] = field(default_factory=dict)  # finished test -> (outcome, message)
    crashed: str | None = None  # the test running when the process died
    finished: bool = False  # the session reached its end


def read_records(lines: list[str]) -> Run:
    run = Run()
    reports: dict[str, list[dict]] = {}
    started = []
    for line in lines:
        record = json.loads(line)
        if record["event"] == "session_finished":
            run.finished = True
        elif record["event"] == "start":
            started.append(record["nodeid"])
        else:
            reports.setdefault(record["nodeid"], []).append(record)
    for test, events in reports.items():
        if not any(event["event"] == "teardown" for event in events):
            continue
        failure = next((event for event in events if event["outcome"] == "failed"), None)
        if failure is not None:
            run.outcomes[test] = ("failed", failure["message"])
        elif any(event["outcome"] == "skipped" for event in events):
            run.outcomes[test] = ("skipped", "")
        else:
            run.outcomes[test] = ("passed", "")
    for k, test in enumerate(started):
        outcome, message = run.outcomes.get(test, ("", ""))
        if outcome == "failed" and CUDA_ERROR.search(message):
            for later in started[k:]:
                run.outcomes.pop(later, None)
            run.crashed, run.finished = test, False
            return run
    unfinished = [test for test in started if test not in run.outcomes]
    if unfinished and not run.finished:
        run.crashed = unfinished[-1]
    return run


def status(outcomes: dict[str, tuple[str, str]], test: str) -> str:
    return outcomes.get(test, ("missing", ""))[0]


def races(plain: dict, delay: dict, confirm: dict) -> list[str]:
    """Tests recorded passing plain and failing both delay runs."""
    return sorted(
        test for test in delay
        if status(plain, test) == "passed" and status(delay, test) == "failed" and status(confirm, test) == "failed")


def entries(found: dict[str, str], repros: dict[str, str], title_prefix: str, job: str) -> list[dict]:
    """One issue entry per test function, listing its failing variants."""
    by_function: dict[str, list[str]] = {}
    for test in sorted(found):
        by_function.setdefault(normalize(test), []).append(test)
    result = []
    for function, tests in by_function.items():
        summary = "Passed without CUTracer and failed twice under CUTracer random delay.\n\n" + "\n\n".join(
            f"{test}\n{found[test]}" for test in tests)
        entry = failure_entry(f"random delay: {function}", title_prefix, job, summary[:MAX_SUMMARY_CHARS],
                              "\n".join(repros[test] for test in tests), False)
        entry["raw_failure_ids"] = "\n".join(tests)
        result.append(entry)
    return result


def show(title: str, log: Path) -> None:
    print(f"::group::{title}", flush=True)
    lines = [line for line in log.read_text(errors="replace").splitlines() if not NOISE.match(line)]
    print("\n".join(lines[-200:]))
    print("::endgroup::", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cutracer-so", required=True)
    parser.add_argument("--kernel", default="", help="Delay only kernels whose name contains this.")
    parser.add_argument("--delay-ns", type=int, default=10000)
    parser.add_argument("--out", type=Path, required=True, help="Scratch: logs, records and CUTracer's files.")
    parser.add_argument("--logs", type=Path, required=True, help="What the run uploads: the delay configs.")
    parser.add_argument("--issue-title-prefix", default=None)
    parser.add_argument("--issue-job", default="sanitize")
    parser.add_argument("--sanitizer-failures", default="", help="The sanitizer step's failures= JSON.")
    parser.add_argument("--known-races", default="", help="JSON: test function -> its open issue number.")
    parser.add_argument("tests", nargs="+")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    args.logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(HERE), os.environ.get("PYTHONPATH")])))

    def pytest(tests: list[str], name: str, delayed: bool) -> Run:
        records, log = args.out / f"{name}.jsonl", args.out / f"{name}.log"
        cmd = PYTEST + tests
        if delayed:
            cmd = [
                "cutracer", "trace", "--cutracer-so", args.cutracer_so, "-a", "random_delay", "--delay-ns",
                str(args.delay_ns), "--delay-dump-path",
                str(args.logs / f"delay-config-{name}.json"), "--output-dir",
                str(args.out)
            ] + (["--kernel-filters", args.kernel] if args.kernel else []) + ["--"] + cmd
        with open(log, "w") as handle:
            subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, env=dict(env, RECORD_OUTCOMES=str(records)),
                           check=False)
        if delayed:
            show(f"{' '.join(tests)} under random delay ({name})", log)
        lines = records.read_text().splitlines() if records.exists() else []
        return read_records(lines)

    known_races: dict[str, int] = json.loads(args.known_races or "{}")
    new_race = False
    found: dict[str, str] = {}
    repros: dict[str, str] = {}
    rows = []
    for i, target in enumerate(args.tests):
        plain_run = pytest([target], f"plain-{i}", False)
        plain = plain_run.outcomes
        order = list(plain)

        delay: dict[str, tuple[str, str]] = {}
        crashed: list[str] = []
        pending, attempt = [target], 0
        while pending:
            run = pytest(pending, f"delay-{i}-{attempt}", True)
            delay.update(run.outcomes)
            if run.crashed:
                crashed.append(run.crashed)
            if run.finished or attempt == MAX_RESUMES:
                break
            # Resume with the tests the plain run recorded that the delay runs have not reached.
            pending = [test for test in order if test not in delay and test not in crashed]
            attempt += 1

        suspects = [test for test in delay if status(delay, test) == "failed" and status(plain, test) == "passed"]
        confirm: dict[str, tuple[str, str]] = {}
        confirm_crashed: list[str] = []
        for k, test in enumerate(suspects[:MAX_CONFIRMS]):
            name = f"confirm-{i}-{k}"
            run = pytest([test], name, True)
            confirm.update(run.outcomes)
            if test not in run.outcomes:
                confirm_crashed.append(test)
            repros[test] = (f"cutracer trace -a random_delay --delay-ns {args.delay_ns} --delay-load-path "
                            f"delay-config-{name}.json -- python -m pytest '{test}'  # config in the run's artifacts")
        confirmed = races(plain, delay, confirm)
        found.update({test: delay[test][1] for test in confirmed})

        not_run = [test for test in order if test not in delay and test not in crashed]
        parts = []
        if not plain_run.finished:
            parts.append("incomplete: the run without CUTracer did not finish")
        if not plain:
            parts.append("not run: the run without CUTracer recorded no test, see its log")
        known = [test for test in confirmed if normalize(test) in known_races]
        new = len(confirmed) - len(known)
        new_race |= new > 0
        if new:
            parts.append(f"race: {new} test(s) failed both delay runs")
        if known:
            issues = ", ".join(sorted({f"#{known_races[normalize(test)]}" for test in known}))
            parts.append(f"known race ({issues}): {len(known)} test(s) failed both delay runs")
        unconfirmed = len(suspects[:MAX_CONFIRMS]) - len(confirmed) - len(confirm_crashed)
        if unconfirmed:
            parts.append(f"{unconfirmed} test(s) failed under delay once but not when rerun alone")
        if len(suspects) > MAX_CONFIRMS:
            parts.append(f"{len(suspects) - MAX_CONFIRMS} test(s) failed under delay once, not rerun (over the cap)")
        plain_failures = [test for test in plain if status(plain, test) == "failed"]
        if plain_failures:
            parts.append(f"{len(plain_failures)} test(s) fail without CUTracer")
        if crashed:
            parts.append(f"crashed under delay (not filed): {', '.join(f'`{test}`' for test in crashed)}")
        if confirm_crashed:
            parts.append(
                f"did not finish when rerun alone (not filed): {', '.join(f'`{test}`' for test in confirm_crashed)}")
        if not_run:
            parts.append(f"incomplete: {len(not_run)} test(s) never ran under delay")
        rows.append(f"| `{target}` | {'; '.join(parts) or 'pass'} |")

    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as handle:
            handle.write("\n### Random delay\n\n| Test | Result |\n|---|---|\n" + "\n".join(rows) + "\n")
    print("\n".join(rows))

    if args.issue_title_prefix:
        failures = json.loads(args.sanitizer_failures or "[]") + entries(found, repros, args.issue_title_prefix,
                                                                         args.issue_job)
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as handle:
                handle.write(f"failures={json.dumps(failures)}\n")
    return int(new_race)


if __name__ == "__main__":
    sys.exit(main())
