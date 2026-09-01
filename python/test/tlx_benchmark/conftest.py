"""pytest options for the perf suite.

Deliberately the same names and choices as the ``bench_<op>.py`` CLI, so the
two entry points are one interface with two front ends rather than two
interfaces that drift.
"""


def pytest_addoption(parser):
    group = parser.getgroup("tlx-benchmark")
    group.addoption(
        "--measure", choices=("latency", "compile", "all"), default="all",
        help="'all' is cheap at --space heuristic (~0.7s cold compile per case); "
        "at --space full the cold pass costs ~4 min PER CASE")
    group.addoption("--space", choices=("full", "heuristic", "smoke"), default="heuristic",
                    help="autotune search space; 'heuristic' is what tlx.ops.mm now uses by default")
    group.addoption("--guard", choices=("off", "report", "enforce"), default="report",
                    help="enforce fails the test on a regression or a compile-cap breach")
    group.addoption("--json", default=None, help="write the machine-readable artifact here")
    group.addoption("--update-baseline", action="store_true",
                    help="record this run as the baseline; refuses noisy and host-bound cases")
    group.addoption("--strict-env", action="store_true",
                    help="fail instead of warning when the environment is not denoised")
