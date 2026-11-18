"""pytest plugin: publish the running test as the step output ``running_test``.

pytest writes no JUnit XML if the process dies. GitHub Actions keeps the last value written for a key, so after a
crash ``running_test`` names the test pytest died in, and it is empty when every test finished.
"""
import os


def _record(nodeid):
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as fh:
            fh.write(f"running_test={nodeid}\n")


def pytest_runtest_logstart(nodeid, location):
    _record(nodeid)


def pytest_runtest_logfinish(nodeid, location):
    _record("")
