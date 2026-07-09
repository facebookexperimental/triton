import gc
from collections import defaultdict
import hashlib

import pytest
import tempfile
import torch


def _top_level_test_key(item):
    nodeid = item.nodeid
    bracket = nodeid.find("[")
    return nodeid if bracket == -1 else nodeid[:bracket]


def _case_key(item):
    return item.name


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def pytest_configure(config):
    # If pytest-sugar is not active, enable instafail
    if not config.pluginmanager.hasplugin("sugar"):
        config.option.instafail = True


@pytest.fixture(autouse=True)
def _gpu_cleanup():
    """Clean up GPU memory between tests to prevent accumulation in bundle mode.

    In bundle mode, all tests in a shard run in a single process. Without
    cleanup, GPU memory from compiled Triton kernels and torch tensors
    accumulates across tests, leading to OOM. This fixture ensures each test
    starts with a clean GPU state.
    """
    yield
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except torch.AcceleratorError:
            # CUDA context may be in an error state after tests that
            # intentionally trigger device-side assertions (e.g. py_debug_test).
            # Silently skip cleanup — the next test will reset the context.
            pass


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")
    parser.addoption(
        "--max-cases-per-test",
        action="store",
        type=int,
        default=100,
        help="Maximum number of cases per top-level test",
    )


def pytest_collection_modifyitems(config, items):
    max_cases = config.getoption("--max-cases-per-test")
    if max_cases <= 0:
        return

    groups = defaultdict(list)
    for item in items:
        groups[_top_level_test_key(item)].append(item)

    kept = []
    deselected = []
    for group in groups.values():
        ordered = sorted(group, key=lambda item: _sha256_hex(_case_key(item)))
        kept.extend(ordered[:max_cases])
        deselected.extend(ordered[max_cases:])

    if deselected:
        config.hook.pytest_deselected(items=deselected)

    items[:] = kept


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture
def fresh_triton_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        from triton import knobs

        with knobs.cache.scope(), knobs.runtime.scope():
            knobs.cache.dir = tmpdir
            yield tmpdir


@pytest.fixture
def fresh_knobs():
    """
    Resets all knobs except ``build``, ``nvidia``, and ``amd`` (preserves
    library paths needed to compile kernels).
    """
    from triton._internal_testing import _fresh_knobs_impl
    fresh_function, reset_function = _fresh_knobs_impl(skipped_attr={"build", "nvidia", "amd"})
    try:
        yield fresh_function()
    finally:
        reset_function()


@pytest.fixture
def fresh_knobs_including_libraries():
    """
    Resets ALL knobs including ``build``, ``nvidia``, and ``amd``.
    Use for tests that verify initial values of these knobs.
    """
    from triton._internal_testing import _fresh_knobs_impl
    fresh_function, reset_function = _fresh_knobs_impl()
    try:
        yield fresh_function()
    finally:
        reset_function()


@pytest.fixture
def with_allocator():
    import triton
    from triton.runtime._allocation import NullAllocator
    from triton._internal_testing import default_alloc_fn

    triton.set_allocator(default_alloc_fn)
    try:
        yield
    finally:
        triton.set_allocator(NullAllocator())
