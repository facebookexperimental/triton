import gc
import pytest
import torch


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
