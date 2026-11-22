"""
Concurrent first use of the LLVM function-list iterator must not abort.

``function_list.__iter__`` builds its iterator via nanobind ``make_iterator``,
which registers the iterator type lazily on first use. Concurrent first use
from multiple compile threads aborted the process in ``nb_type_new``
("concurrently being registered on another thread"): mid-registration,
CPython emits a DeprecationWarning (``builtin type iterator has no
__module__``), whose ``showwarning`` path calls ``os.stat``, which releases
the GIL -- letting a second worker thread enter registration while the
first thread's reservation is still unpublished.

The racy workload runs in a subprocess so a regression fails as a nonzero
exit code instead of aborting the pytest worker. Compile-only, no GPU needed.
"""

import os
import subprocess
import sys
import textwrap

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def _add_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr, OFFSET: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x + OFFSET, mask=mask)


def _compile(block_size, offset):
    target = GPUTarget("hip", "gfx950", 64)
    src = ASTSource(
        _add_kernel,
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "n_elements": "i32"},
        {"BLOCK": block_size, "OFFSET": offset},
    )
    return triton.compile(src, target=target)


def test_function_list_iteration_smoke():
    # Single-threaded compile exercises function_list.__iter__ (the AMD
    # backend iterates get_functions() in make_llir) and must succeed.
    compiled = _compile(block_size=256, offset=0)
    assert "hsaco" in compiled.asm
    assert len(compiled.asm["hsaco"]) > 0


_RACE_SNIPPET = textwrap.dedent(
    """
    import threading

    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource


    @triton.jit
    def _add_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr, OFFSET: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK
        offsets = block_start + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(y_ptr + offsets, x + OFFSET, mask=mask)


    N_THREADS = 16
    barrier = threading.Barrier(N_THREADS)
    errors = []


    def worker(i):
        try:
            # Distinct constexpr per thread so every compile is fresh (no
            # cache hits skipping codegen) and all threads race first use of
            # the function-list iterator type.
            target = GPUTarget("hip", "gfx950", 64)
            src = ASTSource(
                _add_kernel,
                {"x_ptr": "*fp32", "y_ptr": "*fp32", "n_elements": "i32"},
                {"BLOCK": 256, "OFFSET": i},
            )
            barrier.wait(timeout=300)
            triton.compile(src, target=target)
        except Exception as e:  # noqa: BLE001 - surfaced below
            errors.append(repr(e))


    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=600)
    assert not errors, errors
    assert all(not t.is_alive() for t in threads), "worker thread hung"
    """
)


def test_concurrent_function_list_first_use(tmp_path):
    # Fresh compiler cache: cache hits would skip codegen (and the racy
    # iteration) in this or later runs.
    env = {**os.environ, "TRITON_CACHE_DIR": str(tmp_path / "triton-cache")}
    proc = subprocess.run(
        [sys.executable, "-c", _RACE_SNIPPET],
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
    )
    assert proc.returncode == 0, (
        f"concurrent compile workers failed (rc={proc.returncode}):\n{proc.stderr[-4000:]}"
    )
