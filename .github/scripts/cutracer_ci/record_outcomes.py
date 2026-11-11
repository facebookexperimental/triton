"""pytest plugin: append each test's start and reports to ``$RECORD_OUTCOMES``.

One JSON line per event, written as it happens, so a process that dies mid-run
(CUTracer can exit while instrumenting a kernel) keeps the results of the tests
it finished. A ``session_finished`` line marks a run that reached the end.
Load it with ``-p record_outcomes`` and this directory on ``PYTHONPATH``.
"""

import json
import os


def _write(record: dict) -> None:
    with open(os.environ["RECORD_OUTCOMES"], "a") as handle:
        handle.write(json.dumps(record) + "\n")


def pytest_runtest_logstart(nodeid, location):
    _write({"event": "start", "nodeid": nodeid})


def pytest_runtest_logreport(report):
    message = ""
    if report.failed:
        crash = getattr(report.longrepr, "reprcrash", None)
        message = crash.message if crash is not None else str(report.longrepr)[-2000:]
    _write({"event": report.when, "nodeid": report.nodeid, "outcome": report.outcome, "message": message})


def pytest_sessionfinish(session, exitstatus):
    _write({"event": "session_finished"})
