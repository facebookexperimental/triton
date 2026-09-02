def pytest_addoption(parser):
    group = parser.getgroup("tlx-benchmark")
    group.addoption(
        "--space", choices=("heuristic", "full", "smoke"), default="heuristic",
        help="autotune search space; 'heuristic' is what tlx.ops.mm uses by default, and "
        "measuring anything else measures a path users do not take")
    group.addoption("--head", type=int, default=None, metavar="N", help="only the first N cases, for a quick look")
    group.addoption("--synthetic", action="store_true",
                    help="run the correctness shapes instead of this arch's focus list")
    group.addoption("--json", default=None, help="machine-readable artifact (default: bench module's)")
