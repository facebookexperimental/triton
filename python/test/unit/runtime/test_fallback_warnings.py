"""Tests for C-to-Python fallback warnings.

Validates that when TRITON_USE_TRITON_DISPATCHER=1 or TRITON_ENABLE_C_CACHE=1
is set but the C path cannot be used, a visible warning is emitted so users
are aware of the fallback.
"""

import warnings
from unittest.mock import patch

import torch
import triton
import triton.language as tl
from triton import knobs
from triton.compiler.compiler import ASTSource, compile as triton_compile


@triton.jit
def add_kernel(X, Y, OUT, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    y = tl.load(Y + offs, mask=mask)
    tl.store(OUT + offs, x + y, mask=mask)


def _compile_kernel(fn, signature, constexprs=None, attrs=None):
    """Helper to compile a kernel and return the CompiledKernel."""
    target = triton.runtime.driver.active.get_current_target()
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs, attrs=attrs)
    return triton_compile(src, target=target)


class TestDispatcherFallbackWarning:
    """Tests for TRITON_USE_TRITON_DISPATCHER fallback warnings."""

    def test_dispatcher_fallback_warning_in_compiled_kernel_getitem(self, monkeypatch):
        """CompiledKernel.__getitem__ should warn when dispatcher flag is on but _dispatcher is None."""
        monkeypatch.setenv("TRITON_USE_TRITON_DISPATCHER", "1")
        compiled = _compile_kernel(
            add_kernel,
            signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
            constexprs={"BLOCK": 1024},
        )
        compiled._init_handles()
        # Force dispatcher to None to simulate creation failure
        compiled._dispatcher = None

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compiled[(1, 1, 1)]  # noqa: F841
            # Should have emitted a fallback warning
            dispatcher_warnings = [
                x for x in w if "TRITON_USE_TRITON_DISPATCHER=1" in str(x.message)
                and "falling back to Python runner" in str(x.message)
            ]
            assert len(dispatcher_warnings) == 1, (
                f"Expected 1 dispatcher fallback warning, got {len(dispatcher_warnings)}. "
                f"All warnings: {[str(x.message) for x in w]}")

    def test_no_warning_when_dispatcher_flag_off(self, monkeypatch):
        """No warning when TRITON_USE_TRITON_DISPATCHER is not set."""
        monkeypatch.setenv("TRITON_USE_TRITON_DISPATCHER", "0")
        compiled = _compile_kernel(
            add_kernel,
            signature={"X": "*fp32", "Y": "*fp32", "OUT": "*fp32", "N": "i32"},
            constexprs={"BLOCK": 1024},
        )
        compiled._init_handles()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compiled[(1, 1, 1)]  # noqa: F841
            dispatcher_warnings = [x for x in w if "TRITON_USE_TRITON_DISPATCHER" in str(x.message)]
            assert len(dispatcher_warnings) == 0

    def test_dispatcher_fallback_warning_in_jit_run(self, monkeypatch):
        """JITFunction.run() should warn when dispatcher flag is on but kernel has no dispatcher."""
        monkeypatch.setenv("TRITON_USE_TRITON_DISPATCHER", "1")
        monkeypatch.setenv("TRITON_ENABLE_C_CACHE", "0")

        @triton.jit
        def simple_kernel(X, N, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < N
            x = tl.load(X + offs, mask=mask)
            tl.store(X + offs, x + 1, mask=mask)

        N = 1024
        x = torch.randn(N, device="cuda", dtype=torch.float32)

        # First call compiles and populates cache
        simple_kernel[(1, )](x, N, BLOCK=1024)

        # Now patch the kernel cache to remove _dispatcher from the compiled kernel
        # to simulate a fallback scenario
        device = triton.runtime.driver.active.get_current_device()
        kernel_cache = simple_kernel.device_caches[device][0]
        for key, kernel in kernel_cache.items():
            if hasattr(kernel, '_dispatcher'):
                kernel._dispatcher = None

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            simple_kernel[(1, )](x, N, BLOCK=1024)
            dispatcher_warnings = [
                x for x in w if "TRITON_USE_TRITON_DISPATCHER=1" in str(x.message)
                and "falling back to Python launch" in str(x.message)
            ]
            assert len(dispatcher_warnings) == 1, (
                f"Expected 1 dispatcher fallback warning, got {len(dispatcher_warnings)}. "
                f"All warnings: {[str(x.message) for x in w]}")


class TestCCacheFallbackWarning:
    """Tests for TRITON_ENABLE_C_CACHE fallback warnings."""

    def test_c_cache_bypass_warning_with_hooks(self, monkeypatch):
        """C fast path should warn when bypassed due to launch hooks."""
        monkeypatch.setenv("TRITON_ENABLE_C_CACHE", "1")
        monkeypatch.setenv("TRITON_USE_TRITON_DISPATCHER", "0")

        @triton.jit(c_cache=True)
        def hook_kernel(X, N, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < N
            x = tl.load(X + offs, mask=mask)
            tl.store(X + offs, x + 1, mask=mask)

        N = 1024
        x = torch.randn(N, device="cuda", dtype=torch.float32)

        # First call without hooks (compiles and populates cache)
        hook_kernel[(1, )](x, N, BLOCK=1024)

        # Now add a launch_enter_hook to trigger the bypass
        original_hook = knobs.runtime.launch_enter_hook
        try:
            knobs.runtime.launch_enter_hook = lambda *args, **kwargs: None
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                hook_kernel[(1, )](x, N, BLOCK=1024)
                cache_warnings = [
                    x for x in w if "TRITON_ENABLE_C_CACHE" in str(x.message)
                    and "C fast path bypassed" in str(x.message) and "launch_enter_hook active" in str(x.message)
                ]
                assert len(cache_warnings) == 1, (f"Expected 1 C cache bypass warning, got {len(cache_warnings)}. "
                                                  f"All warnings: {[str(x.message) for x in w]}")
        finally:
            knobs.runtime.launch_enter_hook = original_hook

    def test_no_c_cache_warning_when_fast_path_works(self, monkeypatch):
        """No warning should be emitted when C fast path is used successfully."""
        monkeypatch.setenv("TRITON_ENABLE_C_CACHE", "1")
        monkeypatch.setenv("TRITON_USE_TRITON_DISPATCHER", "1")

        @triton.jit(c_cache=True)
        def fast_kernel(X, N, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < N
            x = tl.load(X + offs, mask=mask)
            tl.store(X + offs, x + 1, mask=mask)

        N = 1024
        x = torch.randn(N, device="cuda", dtype=torch.float32)

        # First call compiles
        fast_kernel[(1, )](x, N, BLOCK=1024)

        # Second call should hit C fast path — no warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fast_kernel[(1, )](x, N, BLOCK=1024)
            cache_warnings = [
                x for x in w if "TRITON_ENABLE_C_CACHE" in str(x.message) and "C fast path bypassed" in str(x.message)
            ]
            assert len(cache_warnings) == 0, (f"Expected 0 C cache bypass warnings, got {len(cache_warnings)}. "
                                              f"All warnings: {[str(x.message) for x in w]}")

    def test_c_cache_hit_no_dispatcher_warning(self, monkeypatch):
        """When dispatcher flag is on but kernel has no dispatcher, run() should warn."""
        monkeypatch.setenv("TRITON_ENABLE_C_CACHE", "1")
        monkeypatch.setenv("TRITON_USE_TRITON_DISPATCHER", "1")

        @triton.jit(c_cache=True)
        def nodispatch_kernel(X, N, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < N
            x = tl.load(X + offs, mask=mask)
            tl.store(X + offs, x + 1, mask=mask)

        N = 1024
        x = torch.randn(N, device="cuda", dtype=torch.float32)

        # First call compiles and populates the C cache
        nodispatch_kernel[(1, )](x, N, BLOCK=1024)

        # Remove dispatcher from cached kernels to simulate fallback
        device = triton.runtime.driver.active.get_current_device()
        kernel_cache = nodispatch_kernel.device_caches[device][0]
        for key, kernel in kernel_cache.items():
            if hasattr(kernel, '_dispatcher'):
                kernel._dispatcher = None

        # Force slow path by using _skip_fc=True via callable grid
        # (callable grid bypasses C cache and goes through Python slow path)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            nodispatch_kernel[lambda meta: (1, )](x, N, BLOCK=1024)
            dispatcher_warnings = [
                x for x in w if "TRITON_USE_TRITON_DISPATCHER=1" in str(x.message)
                and "falling back to Python launch" in str(x.message)
            ]
            assert len(dispatcher_warnings) == 1, (
                f"Expected 1 dispatcher fallback warning, got {len(dispatcher_warnings)}. "
                f"All warnings: {[str(x.message) for x in w]}")


class TestProxyFallbackWarning:
    """Tests for JIT proxy creation fallback warning."""

    def test_proxy_creation_failure_warning(self, monkeypatch):
        """Should warn when native_create_jit_proxy returns None."""
        monkeypatch.setenv("TRITON_ENABLE_C_CACHE", "1")
        monkeypatch.setenv("TRITON_USE_TRITON_DISPATCHER", "1")

        @triton.jit(c_cache=True)
        def proxy_kernel(X, N, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < N
            x = tl.load(X + offs, mask=mask)
            tl.store(X + offs, x + 1, mask=mask)

        N = 1024
        x = torch.randn(N, device="cuda", dtype=torch.float32)

        # First call to populate state
        proxy_kernel[(1, )](x, N, BLOCK=1024)

        # Patch native_create_jit_proxy to return None (simulates proxy creation failure)
        with patch("triton.runtime.jit.native_create_jit_proxy", return_value=None):
            # Clear the proxy cache so it tries to create a new proxy
            if hasattr(proxy_kernel, '_jit_proxy_cache'):
                proxy_kernel._jit_proxy_cache = {}

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _ = proxy_kernel[(1, )]
                proxy_warnings = [x for x in w if "C JIT proxy creation returned None" in str(x.message)]
                assert len(proxy_warnings) == 1, (f"Expected 1 proxy fallback warning, got {len(proxy_warnings)}. "
                                                  f"All warnings: {[str(x.message) for x in w]}")
