import gc

import pytest
import tempfile
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "interpreter: indicate whether interpreter supports the test")


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
    from triton._internal_testing import _fresh_knobs_impl
    fresh_function, reset_function = _fresh_knobs_impl()
    try:
        yield fresh_function()
    finally:
        reset_function()


@pytest.fixture
def fresh_knobs_except_libraries():
    """
    A variant of `fresh_knobs` that keeps library path
    information from the environment as these may be
    needed to successfully compile kernels.
    """
    from triton._internal_testing import _fresh_knobs_impl
    fresh_function, reset_function = _fresh_knobs_impl(skipped_attr={"build", "nvidia", "amd"})
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
