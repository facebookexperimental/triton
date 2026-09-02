"""pytest options for the perf suite.

Deliberately the same names and choices as the ``bench_<op>.py`` CLI, so the
two entry points are one interface with two front ends rather than two
interfaces that drift. ``--device`` is absent because pytest owns process
startup and the GPU is already pinned by the time a test runs.
"""


def pytest_addoption(parser):
    group = parser.getgroup("tlx-benchmark")
    group.addoption(
        "--space", choices=("heuristic", "full", "smoke"), default="heuristic",
        help="autotune search space; 'heuristic' is what tlx.ops.mm uses by default, and "
        "measuring anything else measures a path users do not take")
    group.addoption("--head", type=int, default=None, metavar="N", help="only the first N cases, for a quick look")
    group.addoption("--json", default=None, help="machine-readable artifact (default: bench module's)")
