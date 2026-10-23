"""Per-test failures from a GPU job's log, for the /ci/ dashboard.

pytest's short test summary prints one line per failing test,
``FAILED <nodeid> - <message>`` or ``ERROR <nodeid>``, on every trigger. (The
JSON that parse_junit_failures.py prints only exists on scheduled runs.) Each
line goes to the step whose time window holds its timestamp. Ids are normalized
like parse_junit_failures.normalize, so parametrized variants count as one test.
"""
import ast
import re

from ci_health_rules import INFRA_STEP, t

ANSI = re.compile(r"\x1b\[[0-9;]*m")
#: "<timestamp> FAILED|ERROR <nodeid>[ - <message>]". The -v progress line puts the status after the id, so it never matches.
SUMMARY = re.compile(r"^(\S+Z) (FAILED|ERROR) (\S+)(?: - (.*))?$")
PARAM_SUFFIX = re.compile(r"\[.*\]$")


def normalize(node_id):
    return PARAM_SUFFIX.sub("", node_id)


def _iso(ts):
    """GitHub log timestamps carry 7 fractional digits; datetime takes 6."""
    head, _, frac = ts.rstrip("Z").partition(".")
    return f"{head}.{(frac + '000000')[:6]}Z" if frac else ts


def parse_lines(log_text, normalize_ids=True):
    """(timestamp, id, message) for each short-summary line, in log order. Ids are normalized unless
    normalize_ids is False, for a suite whose parameter names what failed (the TLX ops bench's op)."""
    out = []
    for line in log_text.splitlines():
        m = SUMMARY.match(ANSI.sub("", line).rstrip())
        if m:
            out.append((_iso(m.group(1)), normalize(m.group(3)) if normalize_ids else m.group(3), (m.group(4)
                                                                                                   or "").strip()))
    return out


def attach_failures(job, log_text, normalize_ids=True):
    """Set step["failures"] = [{id, msg}] on each failed test step of the job. A failed setup step gets none."""
    lines = [(t(ts), i, msg) for ts, i, msg in parse_lines(log_text, normalize_ids)]
    key = job["name"].split(" / ", 1)[-1]
    for s in job["steps"]:
        if s["conclusion"] != "failure" or INFRA_STEP.search(s["name"]):
            continue
        start, end = t(s.get("started_at")), t(s.get("completed_at"))
        seen = {}
        for ts, i, msg in lines:
            if start and end and start <= ts <= end and i not in seen:
                seen[i] = msg
        s["failures"] = [{"id": i, "msg": m} for i, m in seen.items()] or [{"id": f"{key}/{s['name']}", "msg": ""}]


#: lit's per-test result line, "<timestamp> FAIL: TRITON :: <path> (n of m)", for each test that did not pass.
LIT_RESULT = re.compile(r"^\S+Z (FAIL|XPASS|TIMEOUT|UNRESOLVED): \S+ :: (\S+)")

#: The first diagnostic in a failing test's output block, e.g. "<file>:32:50: error: undefined variable: MMA".
LIT_ERROR = re.compile(r": error: (.*)$")


def attach_lit_failures(job, log_text):
    """Set step["failures"] = [{id: test path, msg}] on the job's failed step when lit reported failing tests; msg
    is the first error in the test's output, else its lit result. A step that failed without any (the build broke
    before lit ran) gets none."""
    seen, cur = {}, None
    for line in log_text.splitlines():
        line = ANSI.sub("", line).rstrip()
        m = LIT_RESULT.match(line)
        if m:
            cur = m.group(2) if m.group(2) not in seen else None
            seen.setdefault(m.group(2), m.group(1))
            continue
        if line.endswith("Z ********************"):  # the end of the test's output block
            cur = None
        e = LIT_ERROR.search(line)
        if cur and e and seen[cur] in ("FAIL", "XPASS", "TIMEOUT", "UNRESOLVED"):
            seen[cur], cur = e.group(1), None
    for s in job["steps"]:
        if s["conclusion"] == "failure" and seen:
            s["failures"] = [{"id": i, "msg": r} for i, r in seen.items()]


SELECTION = re.compile(r"GPU selection for \S+: (\{.*\})")


def parse_selection(log_text):
    """What gpu_test_selection.py chose for a run: {h100, b200, mi350, build, fallback}, or None if not found.
    fallback is True when the changed paths were unavailable and every suite ran."""
    found = None
    for line in log_text.splitlines():
        m = SELECTION.search(ANSI.sub("", line))
        if m:
            found = {k: v is True for k, v in ast.literal_eval(m.group(1)).items()}
    if found is not None:
        found["fallback"] = "Cannot determine changed paths" in log_text
    return found
