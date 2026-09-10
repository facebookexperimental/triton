def pytest_addoption(parser):
    group = parser.getgroup("tlx-benchmark")
    group.addoption(
        "--space", choices=("heuristic", "full", "smoke"), default=None,
        help="autotune search space; the default is each op's own (mm: heuristic, everything "
        "else: full), and measuring anything else measures a path users do not take")
    group.addoption("--head", type=int, default=None, metavar="N",
                    help="only the first N cases per direction, for a quick look")
    group.addoption("--synthetic", action="store_true",
                    help="run the correctness shapes instead of this arch's focus list")
    group.addoption("--fwd-only", action="store_true", dest="fwd_only", help="skip the backward cases")
    group.addoption("--bwd-only", action="store_true", dest="bwd_only", help="skip the forward cases")
    group.addoption(
        "--latency-measure-mode", choices=("wallclock", "gpu_events"), default="wallclock", dest="latency_mode",
        help="'wallclock' (default) times each call as a caller would see it; 'gpu_events' "
        "pre-enqueues the batch behind a blocked stream to isolate device time")
    group.addoption(
        "--cold-compile", choices=("all", "first", "none"), default=None, dest="cold_compile",
        help="how often to time a first call on a fresh cache; the default is each op's own "
        "(mm: all, everything else: first, since their cold pass compiles a full autotune space)")
    group.addoption("--json", default=None, help="machine-readable artifact (default: bench module's)")
