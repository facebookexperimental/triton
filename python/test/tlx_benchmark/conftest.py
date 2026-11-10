import pathlib
import sys

import pytest

# --import-mode=importlib does not put this directory on sys.path for the hooks below.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def pytest_addoption(parser):
    group = parser.getgroup("tlx-benchmark")
    # Not --device: python/test/conftest.py already takes it for the backend name.
    group.addoption("--gpu", default="auto", help="GPU index, or 'auto' (default) for the least-used one")
    group.addoption("--space", choices=("heuristic", "curated", "full", "smoke"), default=None,
                    help="autotune space; migrated providers require heuristic, legacy providers use their op default")
    group.addoption("--head", type=int, default=None, metavar="N",
                    help="only the first N cases per direction, for a quick look")
    group.addoption("--synthetic", action="store_true",
                    help="run the correctness shapes instead of this arch's focus list")
    group.addoption("--fwd-only", action="store_true", dest="fwd_only", help="skip the backward cases")
    group.addoption("--bwd-only", action="store_true", dest="bwd_only", help="skip the forward cases")
    group.addoption("--latency-measure-mode", choices=("wallclock", "gpu_events"), default="gpu_events",
                    dest="latency_mode",
                    help="legacy timing selector; kernel-performance measurements always use verified GPU events")
    group.addoption(
        "--cold-compile", choices=("all", "first", "none"), default=None, dest="cold_compile",
        help="fresh-cache first-call frequency; migrated providers default to all, legacy providers use their op default"
    )
    group.addoption("--json", default=None, help="machine-readable artifact (default: bench module's)")


def pytest_collection_finish(session):
    # Only GPU perf sessions restart under denoise.sh; an xdist worker cannot be re-executed.
    if hasattr(session.config, "workerinput") or not any("governor" in item.fixturenames for item in session.items):
        return
    from _harness.denoise import ensure_denoised, select_device

    ensure_denoised(select_device(session.config.getoption("--gpu")))


@pytest.fixture(scope="session")
def governor(pytestconfig):
    from _harness import driver
    from _harness.denoise import select_device

    device = select_device(pytestconfig.getoption("--gpu"))
    driver.select(device)
    if device is not None:
        device.pin()
    return None
