"""pytest options for the perf suite.

Deliberately the same names and choices as the ``bench_<op>.py`` CLI, so the
two entry points are one interface with two front ends rather than two
interfaces that drift.
"""


def pytest_addoption(parser):
    group = parser.getgroup("tlx-benchmark")
    group.addoption("--measure", choices=("latency", "compile", "all"), default="latency",
                    help="latency is minutes; compile is ~4 min per case at --space full")
    group.addoption("--space", choices=("full", "heuristic", "smoke"), default="full",
                    help="autotune search space; 'full' is what tlx.ops.* uses by default")
    group.addoption("--guard", choices=("off", "report", "enforce"), default="report",
                    help="enforce fails the test on a regression or a compile-cap breach")
    group.addoption("--json", default=None, help="write the machine-readable artifact here")
    group.addoption("--update-baseline", action="store_true",
                    help="record this run as the baseline; refuses noisy and host-bound cases")
    group.addoption("--strict-env", action="store_true",
                    help="fail instead of warning when the environment is not denoised")
