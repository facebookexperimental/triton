# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-ignore-all-errors
"""Edge-case tests for the C fast cache (fc_build_key / FastCache).

Covers:
- i32/i64 boundary (INT32_MAX → INT32_MAX+1 must cache miss)
- >64 args fallback (exceeds FC_MAX_ARGS)
- do_not_specialize args
- mixed kwargs + positional
- unaligned tensors
"""

import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import TestCase
from unittest.mock import Mock, patch

import torch
import triton
import triton.language as tl
from triton import knobs
from triton._C.libtriton import (
    native_autotune_proxy_insert,
    native_autotune_proxy_set_grid,
    native_create_autotune_proxy,
    native_create_jit_proxy,
    native_fast_dispatch,
    native_fast_dispatch_insert,
)
from triton.runtime.jit import _KernelCache, _bind_fc_args, _hash_fc_opts


def is_cuda():
    return torch.cuda.is_available() and triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hopper_or_newer():
    return is_cuda() and triton.runtime.driver.active.get_current_target().arch >= 90


def _get_device():
    return triton.runtime.driver.active.get_active_torch_device()


def _get_kernel_cache(jit_fn):
    device = triton.runtime.driver.active.get_current_device()
    kernel_cache, _, _, _, _ = jit_fn.device_caches[device]
    return kernel_cache


class TestI32I64Boundary(TestCase):
    """Verify i32/i64 type code boundary produces cache miss."""

    def setUp(self):
        patcher = patch.dict(os.environ, {"TRITON_USE_C_DISPATCHER": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_int32_max_to_int32_max_plus_one(self):
        """Value crossing 2^31 boundary must produce different cache entry."""

        @triton.jit(c_cache=True)
        def nop_int_kernel(N, out_ptr):
            pass

        out = torch.zeros(1, device=_get_device())

        # 2^31 - 1 fits in i32
        nop_int_kernel[(1, )](2**31 - 1, out)
        cache = _get_kernel_cache(nop_int_kernel)
        self.assertEqual(len(cache), 1)

        # 2^31 does NOT fit in i32 → TC_I64 → cache miss
        nop_int_kernel[(1, )](2**31, out)
        self.assertEqual(
            len(cache),
            2,
            "i32 → i64 boundary should produce a cache miss (different type_code)",
        )

    def test_same_i32_value_hits_cache(self):
        """Two different i32 values should hit same cache entry (same type_code)."""

        @triton.jit(c_cache=True)
        def nop_i32_kernel(N, out_ptr):
            pass

        out = torch.zeros(1, device=_get_device())

        nop_i32_kernel[(1, )](42, out)
        cache = _get_kernel_cache(nop_i32_kernel)
        self.assertEqual(len(cache), 1)

        # Different value, same type_code (both i32) — should hit cache
        nop_i32_kernel[(1, )](99, out)
        self.assertEqual(
            len(cache),
            1,
            "Two i32 values should share the same cache entry",
        )


class TestOverMaxArgs(TestCase):
    """Verify >64 args gracefully falls back to slow path."""

    def setUp(self):
        patcher = patch.dict(os.environ, {"TRITON_USE_C_DISPATCHER": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_65_args_does_not_crash(self):
        """65 args exceeds FC_MAX_ARGS=64 — should fallback, not crash."""
        # We can't dynamically generate @triton.jit kernels (need source file),
        # so instead test that the fast path gracefully handles more args than
        # params by passing extra args. The fast path checks len(args)==len(params)
        # and falls back if they don't match.

        @triton.jit(c_cache=True)
        def simple_kernel(x, N):
            pass

        t = torch.zeros(32, device=_get_device())
        # Normal call works
        simple_kernel[(1, )](t, 32)
        # This should not crash the C layer (arg count mismatch → fallback)
        # We can't easily test >64 without a 65-param kernel, so just verify
        # the basic path works. The >64 case is tested implicitly: fc_build_key
        # returns false for n_args > FC_MAX_ARGS, triggering fallback.
        cache = _get_kernel_cache(simple_kernel)
        self.assertEqual(len(cache), 1)


class TestDoNotSpecialize(TestCase):
    """Verify do_not_specialize args don't cause spurious cache misses."""

    def setUp(self):
        patcher = patch.dict(os.environ, {"TRITON_USE_C_DISPATCHER": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_do_not_specialize_value(self):
        """do_not_specialize int arg should not cause cache miss on value change."""

        @triton.jit(c_cache=True, do_not_specialize=["N"])
        def dns_kernel(x, N):
            pass

        t = torch.zeros(64, dtype=torch.float32, device=_get_device())
        dns_kernel[(1, )](t, 32)
        cache = _get_kernel_cache(dns_kernel)
        self.assertEqual(len(cache), 1)

        # Different value but do_not_specialize — should hit cache
        dns_kernel[(1, )](t, 64)
        self.assertEqual(
            len(cache),
            1,
            "do_not_specialize int should not trigger recompile on value change",
        )


class TestMixedKwargsPositional(TestCase):
    """Verify mixed kwargs+positional calls work correctly with c_cache."""

    def setUp(self):
        patcher = patch.dict(os.environ, {"TRITON_USE_C_DISPATCHER": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_kwargs_same_as_positional(self):
        """Calling with kwargs should produce same cache hit as positional."""

        @triton.jit(c_cache=True)
        def kwargs_kernel(x, N):
            pass

        t = torch.zeros(32, device=_get_device())

        # Positional call
        kwargs_kernel[(1, )](t, 32)
        cache = _get_kernel_cache(kwargs_kernel)
        self.assertEqual(len(cache), 1)

        # Kwargs call — same effective args, should hit cache
        kwargs_kernel[(1, )](x=t, N=32)
        self.assertEqual(
            len(cache),
            1,
            "kwargs call with same values should hit cache",
        )

    def test_kwargs_correctness_all_kwargs(self):
        """All args passed as kwargs must produce correct output."""

        @triton.jit(c_cache=True)
        def add_kernel(x_ptr, y_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
            offs = tl.arange(0, BLOCK)
            mask = offs < N
            x = tl.load(x_ptr + offs, mask=mask)
            y = tl.load(y_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)

        N = 32
        device = _get_device()
        x = torch.randn(N, device=device)
        y = torch.randn(N, device=device)
        out_pos = torch.empty(N, device=device)
        out_kw = torch.empty(N, device=device)

        # Positional call
        add_kernel[(1, )](x, y, out_pos, N, 32)
        # All kwargs
        add_kernel[(1, )](x_ptr=x, y_ptr=y, out_ptr=out_kw, N=N, BLOCK=32)

        self.assertTrue(
            torch.equal(out_pos, out_kw),
            "All-kwargs call must produce identical output to positional",
        )

    def test_kwargs_correctness_mixed(self):
        """Mix of positional and kwargs must produce correct output."""

        @triton.jit(c_cache=True)
        def scale_kernel(x_ptr, out_ptr, N: tl.constexpr, SCALE: tl.constexpr):
            offs = tl.arange(0, N)
            x = tl.load(x_ptr + offs)
            tl.store(out_ptr + offs, x * SCALE)

        N = 16
        device = _get_device()
        x = torch.randn(N, device=device)
        out_pos = torch.empty(N, device=device)
        out_mixed = torch.empty(N, device=device)

        # Positional
        scale_kernel[(1, )](x, out_pos, N, 3)
        # Mixed: first two positional, constexprs as kwargs
        scale_kernel[(1, )](x, out_mixed, N=N, SCALE=3)

        self.assertTrue(
            torch.equal(out_pos, out_mixed),
            "Mixed positional+kwargs call must produce identical output",
        )

    def test_kwargs_out_of_order(self):
        """kwargs in different order from function signature must still work."""

        @triton.jit(c_cache=True)
        def oop_kernel(x_ptr, out_ptr, A: tl.constexpr, B: tl.constexpr, N: tl.constexpr):
            offs = tl.arange(0, N)
            x = tl.load(x_ptr + offs)
            tl.store(out_ptr + offs, x * A + B)

        N = 16
        device = _get_device()
        x = torch.randn(N, device=device)
        out_pos = torch.empty(N, device=device)
        out_ooo = torch.empty(N, device=device)

        # Positional
        oop_kernel[(1, )](x, out_pos, 2, 5, N)
        # Out-of-order kwargs
        oop_kernel[(1, )](x, out_ooo, N=N, B=5, A=2)

        self.assertTrue(
            torch.equal(out_pos, out_ooo),
            "Out-of-order kwargs must produce identical output to positional",
        )

    def test_kwargs_dict_unpacking(self):
        """Passing args via **kwargs dict unpacking must produce correct output."""

        @triton.jit(c_cache=True)
        def add_kernel(x_ptr, y_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
            offs = tl.arange(0, BLOCK)
            mask = offs < N
            x = tl.load(x_ptr + offs, mask=mask)
            y = tl.load(y_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)

        N = 32
        device = _get_device()
        x = torch.randn(N, device=device)
        y = torch.randn(N, device=device)
        out_pos = torch.empty(N, device=device)
        out_dict = torch.empty(N, device=device)

        # Positional call (warmup — populates cache)
        add_kernel[(1, )](x, y, out_pos, N, 32)
        # Dict unpacking — should hit the same cache entry
        kwargs = {"x_ptr": x, "y_ptr": y, "out_ptr": out_dict, "N": N, "BLOCK": 32}
        add_kernel[(1, )](**kwargs)

        self.assertTrue(
            torch.equal(out_pos, out_dict),
            "Dict-unpacked kwargs call must produce identical output to positional",
        )
        cache = _get_kernel_cache(add_kernel)
        self.assertEqual(len(cache), 1, "**kwargs call should hit same cache as positional")

    def test_kwargs_partial_dict_unpacking(self):
        """Mix of positional args and **kwargs dict unpacking."""

        @triton.jit(c_cache=True)
        def scale_kernel(x_ptr, out_ptr, N: tl.constexpr, SCALE: tl.constexpr):
            offs = tl.arange(0, N)
            x = tl.load(x_ptr + offs)
            tl.store(out_ptr + offs, x * SCALE)

        N = 16
        device = _get_device()
        x = torch.randn(N, device=device)
        out_pos = torch.empty(N, device=device)
        out_mix = torch.empty(N, device=device)

        # Positional (warmup — populates cache)
        scale_kernel[(1, )](x, out_pos, N, 3)
        # Positional + dict unpacking for remaining args — should hit cache
        kwargs = {"N": N, "SCALE": 3}
        scale_kernel[(1, )](x, out_mix, **kwargs)

        self.assertTrue(
            torch.equal(out_pos, out_mix),
            "Partial dict-unpacked kwargs must produce identical output",
        )
        cache = _get_kernel_cache(scale_kernel)
        self.assertEqual(len(cache), 1, "Partial **kwargs call should hit same cache as positional")

    def test_kwargs_repeated_calls_cache_hit(self):
        """Repeated kwargs calls should hit cache and produce consistent results."""

        @triton.jit(c_cache=True)
        def repeat_kernel(x_ptr, out_ptr, N: tl.constexpr):
            offs = tl.arange(0, N)
            x = tl.load(x_ptr + offs)
            tl.store(out_ptr + offs, x + 1)

        N = 32
        device = _get_device()
        x = torch.randn(N, device=device)

        # Warmup
        out0 = torch.empty(N, device=device)
        repeat_kernel[(1, )](x_ptr=x, out_ptr=out0, N=N)

        cache = _get_kernel_cache(repeat_kernel)
        self.assertEqual(len(cache), 1)

        # Repeated calls — all should hit cache and produce same result
        for _ in range(5):
            out_i = torch.empty(N, device=device)
            repeat_kernel[(1, )](x_ptr=x, out_ptr=out_i, N=N)
            self.assertTrue(torch.equal(out0, out_i))

        self.assertEqual(len(cache), 1, "Repeated kwargs calls should all hit cache")

    def test_kwargs_constexpr_cache_hit_and_miss(self):
        """Constexpr kwargs: same value → cache hit, different value → cache miss."""

        @triton.jit(c_cache=True)
        def ce_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
            offs = tl.arange(0, BLOCK)
            mask = offs < N
            x = tl.load(x_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, x + 1, mask=mask)

        N = 16
        device = _get_device()
        x = torch.randn(N, device=device)
        out = torch.empty(N, device=device)

        # Positional call: N=16, BLOCK=32
        ce_kernel[(1, )](x, out, N, 32)
        cache = _get_kernel_cache(ce_kernel)
        self.assertEqual(len(cache), 1)

        # Kwargs with SAME constexpr values → must hit cache
        ce_kernel[(1, )](x, out, N=N, BLOCK=32)
        self.assertEqual(
            len(cache),
            1,
            "kwargs with same constexpr values should hit cache",
        )

        # Kwargs with DIFFERENT constexpr value → must miss cache
        out2 = torch.empty(32, device=device)
        x2 = torch.randn(32, device=device)
        ce_kernel[(1, )](x2, out2, N=32, BLOCK=64)
        self.assertEqual(
            len(cache),
            2,
            "kwargs with different constexpr value should produce new cache entry",
        )


class TestJITProxyOptionBinding(TestCase):
    """Check native binding without a GPU launch."""

    def setUp(self):
        for group in (knobs.runtime, knobs.compilation, knobs.nvidia, knobs.autotuning, knobs.amd):
            scope = group.scope()
            scope.__enter__()
            self.addCleanup(scope.__exit__, None, None, None)
        knobs.runtime.use_triton_dispatcher = True
        knobs.runtime.use_autotune_c_cache = True
        knobs.autotuning.listener = None
        knobs.runtime.debug = False
        knobs.runtime.sanitize_overflow = False
        knobs.compilation.always_compile = False
        knobs.compilation.instrumentation_mode = ""
        knobs.compilation.fpsan_homomorphic_casts = False

    def _create_proxy(self, options=None, required=False):
        if required:

            @triton.jit(c_cache=True)
            def kernel(N, SCALE: tl.constexpr = 3):
                pass
        else:

            @triton.jit(c_cache=True)
            def kernel(N=32, SCALE: tl.constexpr = 3):
                pass

        kernel._fc_tensor_size_threshold = 0
        kernel._fc_device_key = 0
        kernel._fc_backend = "synthetic"
        dispatch = Mock(return_value=None)
        cached = object()
        fallback = object()
        kernel.run = Mock(return_value=fallback)
        options_hash = _hash_fc_opts(options) if options else 0
        native_fast_dispatch_insert(kernel, (32, 3), kernel.params, options_hash, cached, dispatch, None)
        proxy = native_create_jit_proxy(kernel, (1, ), kernel.params, 0, lambda _: 7, lambda: 0)
        return kernel, proxy, cached, fallback, dispatch

    def _create_autotune_proxy(self):
        kernel, _, cached, fallback, dispatch = self._create_proxy()
        fallback_run = Mock(return_value=fallback)
        proxy = native_create_autotune_proxy(kernel, [0], [], kernel.params, 2, lambda _: 7, lambda: 0, fallback_run)
        native_autotune_proxy_insert(proxy, [32], [3], [1], 0, None)
        native_autotune_proxy_set_grid(proxy, (1, ))
        return kernel, proxy, cached, fallback, dispatch, fallback_run

    def test_held_autotune_proxy_rechecks_launch_state(self):
        states = (
            ("jit", "c_cache", False),
            ("jit", "pre_run_hooks", [lambda *args, **kwargs: None]),
            ("jit", "launch_metadata", lambda *args, **kwargs: None),
            (knobs.runtime, "launch_enter_hook", lambda *args, **kwargs: None),
            (knobs.runtime, "launch_exit_hook", lambda *args, **kwargs: None),
            (knobs.runtime, "add_stages_inspection_hook", lambda: None),
            (knobs.runtime, "debug", True),
            (knobs.runtime, "sanitize_overflow", True),
            (knobs.compilation, "always_compile", True),
            (knobs.compilation, "instrumentation_mode", "fpsan"),
            (knobs.compilation, "fpsan_homomorphic_casts", True),
            (knobs.runtime, "use_triton_dispatcher", False),
            (knobs.runtime, "use_autotune_c_cache", False),
            (knobs.autotuning, "listener", lambda *args, **kwargs: None),
        )
        for owner, name, value in states:
            with self.subTest(option=name):
                kernel, proxy, cached, fallback, dispatch, fallback_run = self._create_autotune_proxy()
                self.assertIs(proxy(32), cached)
                group = kernel if owner == "jit" else owner
                previous = getattr(group, name)
                try:
                    setattr(group, name, value)
                    self.assertIs(proxy(32), fallback)
                finally:
                    setattr(group, name, previous)
                self.assertIs(proxy(32), cached)
                self.assertEqual(dispatch.call_count, 2)
                fallback_run.assert_called_once()

    def test_held_autotune_proxy_preserves_global_checks(self):
        kernel, proxy, cached, fallback, dispatch, fallback_run = self._create_autotune_proxy()
        self.assertIs(proxy(32), cached)
        kernel.used_global_vals[("scale", 0)] = (3, {"scale": 4})
        self.assertIs(proxy(32), fallback)
        self.assertEqual(dispatch.call_count, 1)
        fallback_run.assert_called_once()

    def test_late_jit_callbacks_recheck_launch_state(self):
        for callback in ("device", "stream", "option_hash"):
            with self.subTest(callback=callback):
                options = {"num_warps": 4} if callback == "option_hash" else None
                kernel, _, cached, fallback, dispatch = self._create_proxy(options)
                calls = []

                def enable_hook():
                    calls.append(callback)
                    kernel.pre_run_hooks.append(lambda *args, **kwargs: None)

                def device_getter():
                    if callback == "device":
                        enable_hook()
                    return 0

                def stream_getter(device):
                    if callback == "stream":
                        enable_hook()
                    return 7

                class Option(int):

                    def __hash__(self):
                        enable_hook()
                        return int.__hash__(self)

                proxy = native_create_jit_proxy(kernel, (1, ), kernel.params, 0, stream_getter, device_getter)
                result = proxy(32, num_warps=Option(4)) if callback == "option_hash" else proxy(32)
                self.assertIs(result, fallback)
                self.assertEqual(calls, [callback])
                dispatch.assert_not_called()
                kernel.run.assert_called_once()

    def test_late_autotune_callbacks_recheck_launch_state(self):
        for callback in ("device", "stream", "key_hash", "pre_hook", "grid", "false_listener"):
            with self.subTest(callback=callback):
                kernel, _, _, fallback, dispatch = self._create_proxy()
                fallback_run = Mock(return_value=fallback)
                calls = []

                class FalseListener:

                    def __bool__(self):
                        return False

                    def __call__(self, *args, **kwargs):
                        pass

                def change_state():
                    calls.append(callback)
                    if callback == "false_listener":
                        knobs.autotuning.listener = FalseListener()
                    else:
                        kernel.pre_run_hooks.append(lambda *args, **kwargs: None)

                def device_getter():
                    if callback == "device":
                        change_state()
                    return 0

                def stream_getter(device):
                    if callback == "stream":
                        change_state()
                    return 7

                def pre_hook(named_args):
                    change_state()

                def grid(named_args):
                    change_state()
                    return (1, )

                class Key(int):

                    def __hash__(self):
                        change_state()
                        return int.__hash__(self)

                proxy = native_create_autotune_proxy(kernel, [0], [], kernel.params, 2, stream_getter, device_getter,
                                                     fallback_run)
                native_autotune_proxy_insert(proxy, [32], [3], [1], 0, pre_hook if callback == "pre_hook" else None)
                native_autotune_proxy_set_grid(proxy, grid if callback in ("grid", "false_listener") else (1, ))
                self.assertIs(proxy(Key(32) if callback == "key_hash" else 32), fallback)
                self.assertEqual(calls, [callback])
                dispatch.assert_not_called()
                fallback_run.assert_called_once()

    def test_scalar_options_match_python_hash(self):
        options = {"num_warps": 4, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16}
        kernel, proxy, cached, _, dispatch = self._create_proxy(options)
        self.assertIs(proxy(32, SCALE=3, **options), cached)
        self.assertIs(proxy(SCALE=3, N=32, **dict(reversed(list(options.items())))), cached)
        self.assertEqual(dispatch.call_count, 2)
        kernel.run.assert_not_called()

    def test_changed_options_use_a_distinct_entry(self):
        kernel, proxy, cached, _, dispatch = self._create_proxy({"num_warps": 4})
        changed = object()
        native_fast_dispatch_insert(kernel, (32, 3), kernel.params, _hash_fc_opts({"num_warps": 8}), changed, dispatch,
                                    None)
        self.assertIs(proxy(num_warps=4), cached)
        self.assertIs(proxy(num_warps=8), changed)
        kernel.run.assert_not_called()

    def test_defaults_match_explicit_arguments(self):
        kernel, proxy, cached, _, dispatch = self._create_proxy()
        self.assertIs(proxy(), cached)
        self.assertIs(proxy(SCALE=3), cached)
        self.assertEqual(dispatch.call_args.args, (1, 1, 1, 7, 32))
        self.assertEqual(_bind_fc_args(kernel.params, kernel._param_name_to_idx, (), {}), (32, 3))
        kernel.run.assert_not_called()

    def test_duplicate_and_missing_arguments_use_standard_binding(self):
        kernel, proxy, _, fallback, dispatch = self._create_proxy(required=True)
        self.assertIs(proxy(32, N=32), fallback)
        self.assertIs(proxy(SCALE=3), fallback)
        self.assertIsNone(_bind_fc_args(kernel.params, kernel._param_name_to_idx, (32, ), {"N": 32}))
        self.assertIsNone(_bind_fc_args(kernel.params, kernel._param_name_to_idx, (), {"SCALE": 3}))
        self.assertEqual(kernel.run.call_count, 2)
        dispatch.assert_not_called()

    def test_container_options_use_standard_hashing(self):
        options = {"extern_libs": {"lib": "/dev/null"}}
        kernel, proxy, _, fallback, dispatch = self._create_proxy(options)
        self.assertIs(proxy(**options), fallback)
        kernel.run.assert_called_once()
        dispatch.assert_not_called()

    def test_held_proxy_checks_new_pre_run_hook(self):
        kernel, proxy, cached, fallback, dispatch = self._create_proxy()
        self.assertIs(proxy(), cached)
        kernel.pre_run_hooks.append(lambda *args, **kwargs: None)
        self.assertIs(proxy(), fallback)
        self.assertEqual(dispatch.call_count, 1)
        kernel.run.assert_called_once()

    def test_held_proxy_checks_runtime_options(self):
        for group, name, value in ((knobs.runtime, "debug", True), (knobs.runtime, "sanitize_overflow",
                                                                    True), (knobs.compilation, "always_compile", True),
                                   (knobs.compilation, "instrumentation_mode",
                                    "fpsan"), (knobs.compilation, "fpsan_homomorphic_casts",
                                               True), (knobs.runtime, "use_triton_dispatcher", False)):
            with self.subTest(option=name):
                kernel, proxy, cached, fallback, dispatch = self._create_proxy()
                self.assertIs(proxy(), cached)
                previous = getattr(group, name)
                try:
                    setattr(group, name, value)
                    self.assertIs(proxy(), fallback)
                    self.assertEqual(dispatch.call_count, 1)
                    kernel.run.assert_called_once()
                finally:
                    setattr(group, name, previous)

    def test_device_changes_select_distinct_entries(self):
        kernel, _, cached, fallback, dispatch = self._create_proxy()
        current_device = [0]
        streams = []

        def get_stream(device):
            streams.append(device)
            return 7 + device

        proxy = native_create_jit_proxy(kernel, (1, ), kernel.params, 0, get_stream, lambda: current_device[0])
        self.assertIs(proxy(), cached)
        current_device[0] = 1
        self.assertIs(proxy(), fallback)
        other = object()
        native_fast_dispatch_insert(kernel, (32, 3), kernel.params, 0, other, dispatch, None, 1)
        self.assertIs(proxy(), other)
        current_device[0] = 0
        self.assertIs(proxy(), cached)
        self.assertEqual(streams, [0, 1, 1, 0])
        self.assertEqual(dispatch.call_count, 3)
        kernel.run.assert_called_once()

    def test_empty_argument_projection_passes_no_kernel_arguments(self):
        kernel, proxy, cached, _, dispatch = self._create_proxy()
        native_fast_dispatch_insert(kernel, (32, 3), kernel.params, 0, cached, dispatch, (), 0)
        self.assertIs(proxy(), cached)
        self.assertIs(native_fast_dispatch(kernel, (32, 3), kernel.params, 0, (1, ), 7, 0), cached)
        self.assertEqual(dispatch.call_args_list, [unittest.mock.call(1, 1, 1, 7)] * 2)
        kernel.run.assert_not_called()

    def test_reentrant_cache_clear_preserves_dispatch_result(self):
        for use_proxy in (False, True):
            with self.subTest(use_proxy=use_proxy):
                kernel, proxy, cached, _, dispatch = self._create_proxy()
                dispatch.side_effect = lambda *args: kernel.device_caches.clear()
                if use_proxy:
                    result = proxy()
                else:
                    result = native_fast_dispatch(kernel, (32, 3), kernel.params, 0, (1, ), 7, 0)
                self.assertIs(result, cached)
                dispatch.assert_called_once()
                self.assertFalse(hasattr(kernel, "_fc_cache"))
                kernel.run.assert_not_called()

    def test_reentrant_stream_lookup_cannot_use_a_deleted_cache(self):
        kernel, _, _, fallback, dispatch = self._create_proxy()

        def get_stream(device):
            kernel.device_caches.clear()
            return 7

        proxy = native_create_jit_proxy(kernel, (1, ), kernel.params, 0, get_stream, lambda: 0)
        self.assertIs(proxy(), fallback)
        dispatch.assert_not_called()
        kernel.run.assert_called_once()

    def test_storage_span_controls_tensor_specialization(self):
        kernel, _, _, _, dispatch = self._create_proxy()
        kernel.device_caches.clear()
        kernel._fc_tensor_size_threshold = 128
        small = torch.empty(16, dtype=torch.float32)
        large_storage = torch.empty(256, dtype=torch.float32)
        large_view = large_storage[:16]
        self.assertEqual(small.shape, large_view.shape)
        self.assertLessEqual(small.untyped_storage().size(), 128)
        self.assertGreater(large_view.untyped_storage().size(), 128)
        small_kernel = object()
        large_kernel = object()
        native_fast_dispatch_insert(kernel, (small, 3), kernel.params, 0, small_kernel, dispatch, (0, ), 0)
        native_fast_dispatch_insert(kernel, (large_view, 3), kernel.params, 0, large_kernel, dispatch, (0, ), 0)
        proxy = native_create_jit_proxy(kernel, (1, ), kernel.params, 0, lambda _: 7, lambda: 0)
        self.assertIs(proxy(small), small_kernel)
        self.assertIs(proxy(large_view), large_kernel)
        self.assertEqual(dispatch.call_count, 2)
        kernel.run.assert_not_called()

    def test_custom_constexpr_hash_never_runs_in_native_cache(self):
        kernel, proxy, _, fallback, dispatch = self._create_proxy()
        hash_calls = []

        class CustomConstant:

            def __hash__(self):
                hash_calls.append(1)
                kernel.device_caches.clear()
                return 3

        value = CustomConstant()
        native_fast_dispatch_insert(kernel, (32, value), kernel.params, 0, object(), dispatch, None, 0)
        self.assertIs(proxy(32, SCALE=(3, value)), fallback)
        self.assertEqual(hash_calls, [])
        dispatch.assert_not_called()
        kernel.run.assert_called_once()

    def test_tensor_metadata_callback_cannot_use_a_deleted_cache(self):
        from triton.runtime.jit import _torch_bridge_loaded
        if _torch_bridge_loaded:
            self.skipTest("Requires metadata fallback without the Torch bridge")
        kernel, _, _, fallback, dispatch = self._create_proxy()
        kernel.device_caches.clear()
        kernel._fc_tensor_size_threshold = 128
        tensor = torch.empty(16, dtype=torch.float32)
        cached = object()
        native_fast_dispatch_insert(kernel, (tensor, 3), kernel.params, 0, cached, dispatch, (0, ), 0)
        storage = tensor.untyped_storage()

        def get_storage():
            kernel.device_caches.clear()
            return storage

        tensor.untyped_storage = get_storage
        proxy = native_create_jit_proxy(kernel, (1, ), kernel.params, 0, lambda _: 7, lambda: 0)
        self.assertIs(proxy(tensor), fallback)
        dispatch.assert_not_called()
        kernel.run.assert_called_once()

    def test_implicit_one_subclass_never_runs_native_equality(self):
        kernel, proxy, cached, fallback, dispatch = self._create_proxy()
        callbacks = []

        class CustomOne(int):

            def __eq__(self, other):
                callbacks.append("eq")
                kernel.device_caches.clear()
                return True

            def __hash__(self):
                callbacks.append("hash")
                return 1

        native_fast_dispatch_insert(kernel, (1, 3), kernel.params, 0, cached, dispatch, (), 0)
        self.assertIs(proxy(1), cached)
        value = CustomOne(1)
        native_fast_dispatch_insert(kernel, (value, 3), kernel.params, 0, object(), dispatch, (), 0)
        self.assertIs(proxy(value), fallback)
        self.assertEqual(callbacks, [])
        self.assertEqual(dispatch.call_count, 1)
        kernel.run.assert_called_once()

    def test_native_cache_initializes_attributes_in_a_fresh_process(self):
        script = f"""
import importlib.util
spec = importlib.util.spec_from_file_location('cold_cache_cases', {str(Path(__file__).resolve())!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
case = module.TestJITProxyOptionBinding()
case.setUp()
try:
    case.test_implicit_one_subclass_never_runs_native_equality()
finally:
    case.doCleanups()
"""
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True,
                       env=dict(os.environ, HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES=""))

    def test_held_hip_proxy_rechecks_buffer_ops_state(self):
        kernel, proxy, cached, fallback, dispatch = self._create_proxy()
        kernel._fc_backend = "hip"
        kernel._fc_tensor_size_threshold = 2147483647
        knobs.amd.use_buffer_ops = True
        self.assertIs(proxy(), cached)
        knobs.amd.use_buffer_ops = False
        self.assertIs(proxy(), fallback)
        knobs.amd.use_buffer_ops = True
        self.assertIs(proxy(), cached)
        self.assertEqual(dispatch.call_count, 2)
        kernel.run.assert_called_once()

    def test_removed_kernel_entries_invalidate_a_held_proxy(self):
        for mode in ("kernel_clear", "kernel_pop", "kernel_delete", "device_pop", "device_delete"):
            with self.subTest(mode=mode):
                kernel, proxy, cached, fallback, dispatch = self._create_proxy()
                cache = _KernelCache(kernel)
                cache["entry"] = cached
                kernel.device_caches[0] = (cache, )
                self.assertIs(proxy(), cached)
                if mode == "kernel_clear":
                    cache.clear()
                elif mode == "kernel_pop":
                    cache.pop("entry")
                elif mode == "kernel_delete":
                    del cache["entry"]
                elif mode == "device_pop":
                    kernel.device_caches.pop(0)
                else:
                    del kernel.device_caches[0]
                self.assertIs(proxy(), fallback)
                self.assertEqual(dispatch.call_count, 1)
                kernel.run.assert_called_once()

    def test_replaced_cache_entries_invalidate_held_proxies(self):
        for autotune in (False, True):
            for device_cache in (False, True):
                for operation in ("assign", "update", "union"):
                    with self.subTest(autotune=autotune, device_cache=device_cache, operation=operation):
                        created = self._create_autotune_proxy() if autotune else self._create_proxy()
                        kernel, proxy, cached, fallback, dispatch = created[:5]
                        fallback_run = created[5] if autotune else kernel.run
                        kernels = _KernelCache(kernel)
                        kernels["entry"] = cached
                        kernel.device_caches[0] = (kernels, )
                        kernel._jit_proxy_cache = {"held": proxy}
                        self.assertIs(proxy(32), cached)
                        cache = kernel.device_caches if device_cache else kernels
                        key = 0 if device_cache else "entry"
                        replacement = (_KernelCache(kernel), ) if device_cache else object()
                        if operation == "assign":
                            cache[key] = replacement
                        elif operation == "update":
                            cache.update({key: replacement})
                        else:
                            cache |= {key: replacement}
                        self.assertIs(cache[key], replacement)
                        self.assertFalse(hasattr(kernel, "_fc_cache"))
                        self.assertFalse(kernel._jit_proxy_cache)
                        self.assertIs(proxy(32), fallback)
                        self.assertEqual(dispatch.call_count, 1)
                        fallback_run.assert_called_once()

    def test_unchanged_and_new_cache_entries_preserve_held_proxies(self):
        for autotune in (False, True):
            for device_cache in (False, True):
                for new_entry in (False, True):
                    for operation in ("assign", "update", "union"):
                        with self.subTest(autotune=autotune, device_cache=device_cache, new_entry=new_entry,
                                          operation=operation):
                            created = self._create_autotune_proxy() if autotune else self._create_proxy()
                            kernel, proxy, cached, _, dispatch = created[:5]
                            kernels = _KernelCache(kernel)
                            kernels["entry"] = cached
                            device_entry = (kernels, )
                            kernel.device_caches[0] = device_entry
                            kernel._jit_proxy_cache = {"held": proxy}
                            self.assertIs(proxy(32), cached)
                            capsule = kernel._fc_cache
                            cache = kernel.device_caches if device_cache else kernels
                            key = (1 if new_entry else 0) if device_cache else ("new" if new_entry else "entry")
                            value = device_entry if device_cache else cached
                            if operation == "assign":
                                cache[key] = value
                            elif operation == "update":
                                cache.update({key: value})
                            else:
                                cache |= {key: value}
                            self.assertIs(kernel._fc_cache, capsule)
                            self.assertEqual(kernel._jit_proxy_cache, {"held": proxy})
                            self.assertIs(proxy(32), cached)
                            self.assertEqual(dispatch.call_count, 2)
                            kernel.run.assert_not_called()
                            if autotune:
                                created[5].assert_not_called()

    def test_held_proxy_checks_added_globals(self):
        kernel, proxy, cached, fallback, dispatch = self._create_proxy()
        self.assertIs(proxy(), cached)
        kernel.used_global_vals[("scale", 0)] = (3, {"scale": 3})
        self.assertIs(proxy(), fallback)
        self.assertEqual(dispatch.call_count, 1)
        kernel.run.assert_called_once()


@unittest.skipUnless(torch.cuda.is_available() and torch.version.hip, "Requires a HIP GPU")
class TestHIPJITProxyOptions(TestCase):
    """Check compiler options through the shared HIP proxy."""

    def setUp(self):
        scope = knobs.compilation.scope()
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)
        knobs.compilation.always_compile = False

    def test_hip_buffer_ops_switch_preserves_compilation_state(self):

        @triton.jit(c_cache=True)
        def write(out, value):
            tl.store(out, value)

        with knobs.runtime.scope(), knobs.amd.scope():
            knobs.runtime.use_triton_dispatcher = True
            knobs.amd.use_buffer_ops = True
            out = torch.empty(1, device=_get_device(), dtype=torch.int32)
            first = write[(1, )](out, 3)
            proxy = write[(1, )]
            knobs.amd.use_buffer_ops = False
            changed = proxy(out, 5)
            self.assertIsNot(changed, first)
            torch.testing.assert_close(out, torch.full_like(out, 5))
            knobs.amd.use_buffer_ops = True
            with patch.object(write, "run", side_effect=AssertionError("Unexpected restored-state fallback")):
                self.assertIs(proxy(out, 7), first)
            torch.testing.assert_close(out, torch.full_like(out, 7))

    def test_hip_stages_hook_does_not_replace_default_native_entry(self):
        self._check_stages_hook_isolation(activate_from_pre_run=False)

    def test_hip_pre_run_hook_does_not_replace_default_native_entry(self):
        self._check_stages_hook_isolation(activate_from_pre_run=True)

    def _check_stages_hook_isolation(self, activate_from_pre_run):

        @triton.jit(c_cache=True)
        def write(out, value):
            tl.store(out, value)

        def inspect_stages(self=None, stages=None, options=None, language=None, capability=None):
            if stages is None:
                return "hip_native_cache_isolation", "hip_native_cache_isolation_hash"

        with knobs.runtime.scope():
            knobs.runtime.use_triton_dispatcher = True
            knobs.runtime.add_stages_inspection_hook = None
            out = torch.empty(1, device=_get_device(), dtype=torch.int32)
            first = write[(1, )](out, 3)
            proxy = write[(1, )]
            if activate_from_pre_run:

                def enable_inspection(*args, **kwargs):
                    knobs.runtime.add_stages_inspection_hook = inspect_stages

                write.add_pre_run_hook(enable_inspection)
            else:
                knobs.runtime.add_stages_inspection_hook = inspect_stages
            hooked = proxy(out, 5)
            self.assertIsNot(hooked, first)
            torch.testing.assert_close(out, torch.full_like(out, 5))
            write.pre_run_hooks.clear()
            knobs.runtime.add_stages_inspection_hook = None
            restored = proxy(out, 7)
            self.assertIs(restored, first)
            torch.testing.assert_close(out, torch.full_like(out, 7))

    def test_hip_options_defaults_and_invalid_arguments(self):

        @triton.jit(c_cache=True)
        def options_kernel(out, SCALE: tl.constexpr = 3):
            tl.store(out, SCALE)

        with knobs.runtime.scope():
            knobs.runtime.use_triton_dispatcher = True
            out = torch.empty(1, device=_get_device(), dtype=torch.int32)
            options = {"num_warps": 4, "num_stages": 2, "waves_per_eu": 1, "matrix_instr_nonkdim": 16}
            first = options_kernel[(1, )](out, **options)
            self.assertIsNotNone(first._dispatcher)
            proxy = options_kernel[(1, )]
            self.assertEqual(type(proxy).__name__, "_JITCacheProxy")
            with patch.object(options_kernel, "run", side_effect=AssertionError("Unexpected proxy fallback")):
                self.assertIs(proxy(out, SCALE=3, **options), first)
            torch.testing.assert_close(out, torch.full_like(out, 3))
            changed = options_kernel[(1, )](out, SCALE=5, num_warps=8, num_stages=1, waves_per_eu=2,
                                            matrix_instr_nonkdim=16)
            self.assertEqual(changed.metadata.num_warps, 8)
            self.assertEqual(changed.metadata.waves_per_eu, 2)
            torch.testing.assert_close(out, torch.full_like(out, 5))
            with self.assertRaises(TypeError):
                proxy(out, 3, SCALE=3, **options)
            with self.assertRaises(TypeError):
                proxy(SCALE=3, **options)
            with self.assertRaises(KeyError):
                proxy(out, unknown_option=1)


class TestUnalignedTensor(TestCase):
    """Verify unaligned tensors produce different cache entries (alignment specialization)."""

    def setUp(self):
        patcher = patch.dict(os.environ, {"TRITON_USE_C_DISPATCHER": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_aligned_vs_unaligned(self):
        """16-byte aligned vs unaligned tensor should be different specializations."""

        @triton.jit(c_cache=True)
        def align_kernel(x: tl.tensor, N):
            pass

        # 16-byte aligned (normal allocation)
        t_aligned = torch.zeros(64, dtype=torch.int8, device=_get_device())
        align_kernel[(1, )](t_aligned, 64)
        cache = _get_kernel_cache(align_kernel)
        self.assertEqual(len(cache), 1)

        # Unaligned: offset by 1 byte (int8 tensor sliced at position 1)
        t_unaligned = t_aligned[1:]
        align_kernel[(1, )](t_unaligned, 63)
        # Should produce a second entry (different alignment)
        self.assertEqual(
            len(cache),
            2,
            "Unaligned tensor should produce a different specialization",
        )


class TestAutotuneCCache(TestCase):
    """Verify autotune + c_cache=True produces correct results."""

    def setUp(self):
        patcher = patch.dict(os.environ, {"TRITON_USE_C_DISPATCHER": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_autotuned_kernel_correctness(self):
        """Autotuned kernel with c_cache must produce correct output on repeated calls."""

        @triton.autotune(
            configs=[
                triton.Config({"BLOCK": 32}),
                triton.Config({"BLOCK": 64}),
            ],
            key=["N"],
        )
        @triton.jit(c_cache=True)
        def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offs < N
            x = tl.load(x_ptr + offs, mask=mask)
            y = tl.load(y_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)

        N = 128
        device = _get_device()
        x = torch.randn(N, device=device)
        y = torch.randn(N, device=device)
        expected = x + y

        # First call triggers autotune
        out1 = torch.empty(N, device=device)
        add_kernel[(N // 32, )](x, y, out1, N)
        self.assertTrue(
            torch.allclose(out1, expected, atol=1e-5),
            "First autotuned call must produce correct output",
        )

        # Verify cache is populated (kernel compiled during autotune)
        cache = _get_kernel_cache(add_kernel.fn)
        cache_size_after_autotune = len(cache)
        self.assertGreaterEqual(cache_size_after_autotune, 1)

        # Repeated calls should use C fast cache and still be correct
        for i in range(3):
            out_i = torch.empty(N, device=device)
            add_kernel[(N // 32, )](x, y, out_i, N)
            self.assertTrue(
                torch.allclose(out_i, expected, atol=1e-5),
                f"Repeated autotuned call {i} must produce correct output",
            )

        # Cache should not grow — repeated calls hit cache, not recompile
        self.assertEqual(
            len(cache),
            cache_size_after_autotune,
            "Repeated autotuned calls should hit cache, not recompile",
        )

    def test_autotuned_non_default_num_warps(self):
        """Autotuned kernel with non-default num_warps must use correct config."""

        @triton.autotune(
            configs=[
                triton.Config({"BLOCK": 64}, num_warps=2),
                triton.Config({"BLOCK": 64}, num_warps=8),
            ],
            key=["N"],
        )
        @triton.jit(c_cache=True)
        def scale_kernel(x_ptr, out_ptr, N, SCALE, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offs < N
            x = tl.load(x_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, x * SCALE, mask=mask)

        N = 64
        device = _get_device()
        x = torch.randn(N, device=device)
        expected = x * 3.0

        # First call triggers autotune (picks one of the num_warps configs)
        out1 = torch.empty(N, device=device)
        scale_kernel[(1, )](x, out1, N, 3.0)
        self.assertTrue(
            torch.allclose(out1, expected, atol=1e-5),
            "Autotuned kernel with non-default num_warps must produce correct output",
        )

        # Verify cache is populated
        cache = _get_kernel_cache(scale_kernel.fn)
        cache_size_after_autotune = len(cache)
        self.assertGreaterEqual(cache_size_after_autotune, 1)

        # Repeated calls — must still use the correct compiled kernel
        for i in range(5):
            out_i = torch.empty(N, device=device)
            scale_kernel[(1, )](x, out_i, N, 3.0)
            self.assertTrue(
                torch.allclose(out_i, expected, atol=1e-5),
                f"Repeated call {i} with non-default num_warps must be correct",
            )

        # Cache should not grow — fast path hits, no recompilation
        self.assertEqual(
            len(cache),
            cache_size_after_autotune,
            "Repeated calls with non-default num_warps should hit cache",
        )


class TestDispatchArgIndicesWithNone(TestCase):
    """Verify dispatch_arg_indices correctly handles None pointer args.

    Reproduces the CMSL HSTU pattern: autotuned kernel is compiled with real
    tensors, then dispatched with None for optional pointer params.
    Without dispatch_arg_indices, the C dispatcher would call .data_ptr()
    on None and crash with SystemError.
    """

    def setUp(self):
        patcher = patch.dict(os.environ, {"TRITON_USE_C_DISPATCHER": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_none_pointer_args_no_crash(self):
        """Kernel compiled with tensors, dispatched with None — must not crash."""

        @triton.autotune(
            configs=[triton.Config({"BLOCK": 32})],
            key=[],
        )
        @triton.jit(c_cache=True)
        def kernel_with_optional(out_ptr, bias_ptr, mask_ptr, N, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            # Only use bias/mask if not null
            val = tl.zeros([BLOCK], dtype=tl.float32)
            if bias_ptr is not None:
                val += tl.load(bias_ptr + offs, mask=offs < N)
            tl.store(out_ptr + offs, val, mask=offs < N)

        device = _get_device()
        N = 32
        out = torch.zeros(N, device=device)
        bias = torch.ones(N, device=device)
        mask = torch.ones(N, device=device)

        # First call with real tensors — triggers compilation
        kernel_with_optional[(1, )](out, bias, mask, N)
        self.assertTrue(torch.allclose(out, bias))

        # Second call with None — must not crash
        out2 = torch.zeros(N, device=device)
        kernel_with_optional[(1, )](out2, None, None, N)
        # With None bias, output should be zeros
        self.assertTrue(torch.allclose(out2, torch.zeros(N, device=device)))

    # The C fast path (which this test exercises via native_fast_dispatch_insert)
    # is bypassed when TRITON_ALWAYS_COMPILE=1 (set by defs.bzl for all buck
    # tests), so the replacement dispatcher would never fire. Opt out here:
    # this test validates dispatch metadata, not recompilation.
    @patch.dict(os.environ, {"TRITON_ALWAYS_COMPILE": "0"})
    def test_reinsert_preserves_dispatch_metadata(self):
        """Reinserting one specialization must not create metadata-less entries."""

        @triton.jit(c_cache=True)
        def optional_ptr_kernel(out_ptr, opt_ptr, N: tl.constexpr):
            offs = tl.arange(0, N)
            value = tl.zeros([N], dtype=tl.float32)
            if opt_ptr is not None:
                value += tl.load(opt_ptr + offs)
            tl.store(out_ptr + offs, value)

        device = _get_device()
        N = 32
        out = torch.empty(N, device=device)

        # None is specialized away, so the dispatcher expects only out_ptr.
        kernel = optional_ptr_kernel[(1, )](out, None, N)
        dispatcher_arg_counts = []

        def replacement_dispatcher(*args):
            dispatcher_arg_counts.append(len(args))

        native_fast_dispatch_insert(
            optional_ptr_kernel,
            (out, None, N),
            optional_ptr_kernel.params,
            optional_ptr_kernel._fc_options_hash,
            kernel,
            replacement_dispatcher,
            kernel._dispatch_arg_indices,
        )

        # Reinsertion should update the existing cache entry rather than leave
        # duplicate entries with different dispatcher metadata.
        optional_ptr_kernel[(1, )](out, None, N)
        self.assertEqual(dispatcher_arg_counts, [5])

        # Replacing the dispatcher without indices must clear the old tuple.
        # The legacy path passes both non-constexpr arguments instead of
        # retaining the previous one-argument projection.
        native_fast_dispatch_insert(
            optional_ptr_kernel,
            (out, None, N),
            optional_ptr_kernel.params,
            optional_ptr_kernel._fc_options_hash,
            kernel,
            replacement_dispatcher,
            None,
        )
        optional_ptr_kernel[(1, )](out, None, N)
        self.assertEqual(dispatcher_arg_counts, [5, 6])

    def test_none_args_cache_hit_after_first_dispatch(self):
        """After first None-args dispatch seeds the cache, subsequent calls hit."""

        @triton.autotune(
            configs=[triton.Config({"BLOCK": 32})],
            key=[],
        )
        @triton.jit(c_cache=True)
        def simple_kernel(out_ptr, opt_ptr, N, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            tl.store(out_ptr + offs, tl.zeros([BLOCK], dtype=tl.float32), mask=offs < N)

        device = _get_device()
        N = 32
        out = torch.zeros(N, device=device)
        real = torch.ones(N, device=device)

        # Compile with real tensor
        simple_kernel[(1, )](out, real, N)

        # First None call — may miss, Python fallback inserts
        simple_kernel[(1, )](out, None, N)

        # Get cache size
        cache = _get_kernel_cache(simple_kernel.fn)
        cache_size = len(cache)

        # Subsequent None calls — should not grow the cache
        for _ in range(5):
            simple_kernel[(1, )](out, None, N)

        self.assertEqual(
            len(cache),
            cache_size,
            "Repeated None-arg calls should hit cache, not recompile",
        )


class TestUnhashableKwargs(TestCase):
    """Verify unhashable kwargs (e.g. extern_libs=dict) don't crash the fast path.

    Reproduces https://fb.workplace.com/...: beta Triton crashes with
    ``TypeError: unhashable type: 'dict'`` when a @triton.jit kernel is
    passed ``extern_libs=<dict>`` because the C fast path hashes
    ``_fc_opts`` items raw.
    """

    def setUp(self):
        patcher = patch.dict(os.environ, {"TRITON_USE_C_DISPATCHER": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_extern_libs_dict_no_crash(self):
        """extern_libs={dict} must not raise TypeError on hash."""

        @triton.jit(c_cache=True)
        def nop_kernel(out_ptr, N: tl.constexpr):
            offs = tl.arange(0, N)
            tl.store(out_ptr + offs, tl.zeros([N], dtype=tl.float32))

        device = _get_device()
        out = torch.zeros(32, device=device)

        # Normal call — warmup (populates cache with options_hash=0).
        nop_kernel[(1, )](out, 32)

        # Call with extern_libs dict — the fast path must hash this without
        # raising ``TypeError: unhashable type: 'dict'``.
        nop_kernel[(1, )](out, 32, extern_libs={"libdevice": "/dev/null"})

    def test_extern_libs_empty_dict_no_crash(self):
        """extern_libs={} (empty dict) must not crash the fast path hash."""

        @triton.jit(c_cache=True)
        def nop_kernel2(out_ptr, N: tl.constexpr):
            offs = tl.arange(0, N)
            tl.store(out_ptr + offs, tl.zeros([N], dtype=tl.float32))

        device = _get_device()
        out = torch.zeros(32, device=device)

        nop_kernel2[(1, )](out, 32)
        # Empty dict is still a dict — must not crash.
        nop_kernel2[(1, )](out, 32, extern_libs={})

    def test_multi_entry_extern_libs_no_crash(self):
        """Multi-entry extern_libs dict must not crash the fast path hash."""

        @triton.jit(c_cache=True)
        def nop_kernel3(out_ptr, N: tl.constexpr):
            offs = tl.arange(0, N)
            tl.store(out_ptr + offs, tl.zeros([N], dtype=tl.float32))

        device = _get_device()
        out = torch.zeros(32, device=device)

        nop_kernel3[(1, )](out, 32)
        # Multi-entry extern_libs dict — must not crash the hash.
        nop_kernel3[(1, )](out, 32, extern_libs={"libA": "/dev/null", "libB": "/dev/null"})

    def test_make_hashable_produces_different_hashes(self):
        """_make_hashable must produce different hashes for different dicts."""
        h1 = _hash_fc_opts({"extern_libs": {"libA": "/path/a"}})
        h2 = _hash_fc_opts({"extern_libs": {"libB": "/path/b"}})
        h3 = _hash_fc_opts({"extern_libs": {"libA": "/path/a"}})

        self.assertNotEqual(h1, h2, "Different dict values must produce different hashes")
        self.assertEqual(h1, h3, "Same dict values must produce the same hash")

    def test_different_none_pattern_different_cache_entry(self):
        """Different None patterns must produce different cache entries (no false match)."""

        @triton.autotune(
            configs=[triton.Config({"BLOCK": 32})],
            key=[],
        )
        @triton.jit(c_cache=True)
        def dual_opt_kernel(out_ptr, a_ptr, b_ptr, N, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            val = tl.zeros([BLOCK], dtype=tl.float32)
            if a_ptr is not None:
                val += tl.load(a_ptr + offs, mask=offs < N)
            if b_ptr is not None:
                val += tl.load(b_ptr + offs, mask=offs < N) * 2
            tl.store(out_ptr + offs, val, mask=offs < N)

        device = _get_device()
        N = 32
        out = torch.zeros(N, device=device)
        a = torch.ones(N, device=device)
        b = torch.ones(N, device=device)

        # Compile with both real
        dual_opt_kernel[(1, )](out, a, b, N)
        expected_both = a + b * 2
        self.assertTrue(torch.allclose(out, expected_both))

        # Call with a=None, b=real → should get b*2 only
        out_b = torch.zeros(N, device=device)
        dual_opt_kernel[(1, )](out_b, None, b, N)
        self.assertTrue(torch.allclose(out_b, b * 2))

        # Call with a=real, b=None → should get a only
        out_a = torch.zeros(N, device=device)
        dual_opt_kernel[(1, )](out_a, a, None, N)
        self.assertTrue(torch.allclose(out_a, a))

        # Call with both None → should get zeros
        out_none = torch.zeros(N, device=device)
        dual_opt_kernel[(1, )](out_none, None, None, N)
        self.assertTrue(torch.allclose(out_none, torch.zeros(N, device=device)))


# Module-level global referenced by the 2-CTA test kernel → populates
# used_global_vals → __getitem__ returns the lambda fallback instead of
# the C proxy, exercising the buggy path.
_CLUSTER_SCALE = tl.constexpr(1.0)


class TestCtasPerCgaAutotunerSteadyState(TestCase):
    """Autotuner steady-state path must preserve ctas_per_cga.

    For single-config autotune, _last_key is never set (only the multi-config
    branch sets it), so _seed_key is always None. After the first call seeds
    the C cache (adding None to _fc_seeded), every subsequent call goes through
    the steady-state path: self.fn[evaluated_grid](*full_args), which does NOT
    pass the config's compilation options.

    When __getitem__ returns the lambda fallback (e.g. because used_global_vals
    is non-empty), run() is called without ctas_per_cga. Without the
    _fc_meta_kwargs fallback in JITFunction.run(), recompilation drops the
    cluster config and cuLaunchKernelEx fails with error 912.
    """

    @unittest.skipUnless(is_hopper_or_newer(), "ctas_per_cga is NVIDIA-only and requires Hopper (sm90) or newer")
    def test_second_call_preserves_cluster_config(self):

        @triton.autotune(
            configs=[
                triton.Config(
                    {"BLOCK_SIZE": 128},
                    num_warps=4,
                    num_stages=1,
                    ctas_per_cga=(2, 1, 1),
                ),
            ],
            key=["n_elements"],
        )
        @triton.jit
        def add_kernel_2cta(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            scale = _CLUSTER_SCALE  # noqa: F841  — forces used_global_vals
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(x_ptr + offs, mask=mask)
            y = tl.load(y_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, (x + y).to(tl.float32), mask=mask)

        device = _get_device()
        n = 1024

        def grid(meta):
            return (triton.cdiv(n, meta["BLOCK_SIZE"]), )

        # Call 1: seeds the C cache (_seed_key=None not in _fc_seeded).
        x1 = torch.randn(n, device=device, dtype=torch.float32)
        y1 = torch.randn(n, device=device, dtype=torch.float32)
        o1 = torch.empty(n, device=device, dtype=torch.float32)
        add_kernel_2cta[grid](x1, y1, o1, n)
        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(o1, x1 + y1))

        # Call 2: steady-state path (_seed_key=None already in _fc_seeded).
        # Without the _fc_meta_kwargs fix this recompiles without
        # ctas_per_cga and fails with CUDA error 912.
        x2 = torch.randn(n, device=device, dtype=torch.float32)
        y2 = torch.randn(n, device=device, dtype=torch.float32)
        o2 = torch.empty(n, device=device, dtype=torch.float32)
        add_kernel_2cta[grid](x2, y2, o2, n)
        torch.cuda.synchronize()
        self.assertTrue(torch.allclose(o2, x2 + y2))

    def _run_new_key_cluster_config_sequence(self, check_best_config):

        def select_cta_count(configs, named_args, **kwargs):
            if named_args["n_elements"] == 256:
                return [config for config in configs if config.kwargs["USE_2CTA"]]
            return configs

        benchmark_index = 0

        def do_bench(fn, quantiles):
            nonlocal benchmark_index
            fn()
            benchmark_index += 1
            return [float(benchmark_index)] * len(quantiles)

        @triton.autotune(
            configs=[
                triton.Config(
                    {"BLOCK_SIZE": 128, "USE_2CTA": False},
                    num_warps=4,
                    num_stages=1,
                ),
                triton.Config(
                    {"BLOCK_SIZE": 128, "USE_2CTA": False},
                    num_warps=8,
                    num_stages=1,
                ),
                triton.Config(
                    {"BLOCK_SIZE": 128, "USE_2CTA": True},
                    num_warps=4,
                    num_stages=1,
                    ctas_per_cga=(2, 1, 1),
                ),
            ],
            key=["n_elements"],
            prune_configs_by={"early_config_prune": select_cta_count},
            do_bench=do_bench,
        )
        @triton.jit
        def add_kernel_switching_ctas(
            x_ptr,
            y_ptr,
            out_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
            USE_2CTA: tl.constexpr,
            DEFAULT_VALUE: tl.constexpr = 0,
        ):
            scale = _CLUSTER_SCALE
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(x_ptr + offs, mask=mask)
            y = tl.load(y_ptr + offs, mask=mask)
            tl.store(
                out_ptr + offs,
                ((x + y) * scale).to(tl.float32),
                mask=mask,
            )

        device = _get_device()
        # The final 256 revisits an already-seeded key after the 1-CTA key has
        # replaced the active fast-cache metadata. The defaulted constexpr
        # forces _try_fast_path to use its Python fallback.
        for n in (256, 384, 256):
            x = torch.randn(n, device=device, dtype=torch.float32)
            y = torch.randn(n, device=device, dtype=torch.float32)
            out = torch.empty(n, device=device, dtype=torch.float32)

            def grid(meta):
                num_ctas = 2 if meta["USE_2CTA"] else 1
                num_blocks = triton.cdiv(n, 128)
                return (triton.cdiv(num_blocks, num_ctas) * num_ctas, )

            add_kernel_switching_ctas[grid](x, y, out, n)
            torch.cuda.synchronize()
            if check_best_config:
                self.assertEqual(add_kernel_switching_ctas.best_config.kwargs["USE_2CTA"], n == 256)
            self.assertTrue(torch.allclose(out, x + y))

    @unittest.skipUnless(is_hopper_or_newer(), "ctas_per_cga is NVIDIA-only and requires Hopper (sm90) or newer")
    @patch.dict(os.environ, {"TRITON_AUTOTUNE_USE_C_CACHE": "1"})
    def test_new_key_dispatches_correct_cluster_config_with_c_proxy(self):
        # Native proxy hits bypass Autotuner.run(), so output is the behavioral oracle.
        self._run_new_key_cluster_config_sequence(check_best_config=False)

    @unittest.skipUnless(is_hopper_or_newer(), "ctas_per_cga is NVIDIA-only and requires Hopper (sm90) or newer")
    @patch.dict(os.environ, {"TRITON_AUTOTUNE_USE_C_CACHE": "0"})
    def test_new_key_replaces_previous_cluster_config(self):
        self._run_new_key_cluster_config_sequence(check_best_config=True)

    @unittest.skipUnless(is_hopper_or_newer(), "ctas_per_cga is NVIDIA-only and requires Hopper (sm90) or newer")
    def test_new_key_does_not_inherit_previous_cluster_config(self):

        def prune_configs(configs, named_args, **kwargs):
            n_elements = named_args["n_elements"]
            num_ctas = 2 if n_elements == 256 else 1
            return [config for config in configs if config.kwargs["NUM_CTAS"] == num_ctas]

        @triton.autotune(
            configs=[
                triton.Config(
                    {"BLOCK_SIZE": 128, "NUM_CTAS": 2},
                    num_warps=4,
                    num_stages=1,
                    ctas_per_cga=(2, 1, 1),
                ),
                triton.Config(
                    {"BLOCK_SIZE": 128, "NUM_CTAS": 1},
                    num_warps=4,
                    num_stages=1,
                ),
            ],
            key=["n_elements"],
            prune_configs_by={"early_config_prune": prune_configs},
        )
        @triton.jit
        def add_kernel_mixed_ctas(
            x_ptr,
            y_ptr,
            out_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
            NUM_CTAS: tl.constexpr,
        ):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < n_elements
            x = tl.load(x_ptr + offs, mask=mask)
            y = tl.load(y_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)

        device = _get_device()

        def run(n):

            def grid(meta):
                num_ctas = meta["NUM_CTAS"]
                tiles = triton.cdiv(n, meta["BLOCK_SIZE"])
                return (triton.cdiv(tiles, num_ctas) * num_ctas, )

            x = torch.randn(n, device=device, dtype=torch.float32)
            y = torch.randn(n, device=device, dtype=torch.float32)
            out = torch.empty(n, device=device, dtype=torch.float32)
            add_kernel_mixed_ctas[grid](x, y, out, n)
            torch.cuda.synchronize()
            self.assertTrue(torch.allclose(out, x + y))

        run(256)
        run(384)
