"""Point-in-time nightly commit selection (pure, unit-tested).

A check is green iff its LATEST completed verdict (success/failure/timed_out)
as of the cut time is success; cancelled runs are ignored.
"""
from __future__ import annotations

_VERDICT = ("success", "failure", "timed_out")


def canonical_check_name(name):
    # Reusable GPU workflows prefix check names with their caller job. Accept
    # both forms so the rollout can still select commits tested before it.
    for platform in ("h100", "b200", "mi350"):
        if name == f"{platform} / {platform}-tlx-test":
            return f"{platform}-tlx-test"
    return name


def check_green(runs, cut):
    verdicts = sorted(
        (x for x in runs if x.get("conclusion") in _VERDICT and x.get("completed_at") and x["completed_at"] <= cut),
        key=lambda x: x["completed_at"],
    )
    return bool(verdicts) and verdicts[-1]["conclusion"] == "success"


def commit_green(runs_by_check, required, cut):
    return all(check_green(runs_by_check.get(name, []), cut) for name in required)


def select(shas, fetch, required, cut, cap=100):
    for sha in shas[:cap]:
        if commit_green(fetch(sha), required, cut):
            return sha
    return None
