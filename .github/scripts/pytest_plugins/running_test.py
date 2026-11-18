"""pytest plugin: keep the node id of the running test in $PYTEST_RUNNING_TEST_FILE.

pytest writes no JUnit XML if the process dies, so this file is what names the test a crash happened in.
"""
import os


def _record(text):
    path = os.environ.get("PYTEST_RUNNING_TEST_FILE")
    if path:
        with open(path, "w") as fh:
            fh.write(text)


def pytest_runtest_logstart(nodeid, location):
    _record(nodeid)


def pytest_runtest_logfinish(nodeid, location):
    _record("")
