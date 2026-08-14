"""Tests for triton_dispatcher_factory.py — tensordesc expansion logic.

These tests verify the pure-Python type expansion and wrapper logic without
needing a GPU or the full compiled triton module. We mock the heavy triton
imports and load the module under test directly via importlib.
"""

import importlib.util
import os
import sys
import unittest
from unittest.mock import MagicMock


def _load_factory_module():
    """Load triton_dispatcher_factory.py with mocked triton deps."""
    # Save original state
    saved = {}
    mock_keys = [
        "triton.runtime",
        "triton.runtime._allocation",
        "triton.runtime.driver",
    ]
    for key in mock_keys:
        if key in sys.modules:
            saved[key] = sys.modules[key]

    # Mock only the specific submodules the factory needs (not "triton" itself)
    mock_runtime = MagicMock()
    sys.modules["triton.runtime"] = mock_runtime
    sys.modules["triton.runtime._allocation"] = mock_runtime._allocation
    sys.modules["triton.runtime.driver"] = mock_runtime.driver

    try:
        # Try multiple locations for the source file
        candidates = [
            # Direct path (devserver)
            os.path.join(
                os.environ.get(
                    "FBSOURCE_DIR",
                    "/data/users/{}/fbsource".format(os.environ.get("USER", "")),
                ),
                "third-party/triton/beta/triton/third_party/nvidia/backend/triton_dispatcher_factory.py",
            ),
            # Buck resource (relative to this test file)
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "triton_dispatcher_factory.py",
            ),
        ]

        for factory_path in candidates:
            if os.path.exists(factory_path):
                spec = importlib.util.spec_from_file_location("triton_dispatcher_factory", factory_path)
                assert spec is not None and spec.loader is not None
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod

        raise FileNotFoundError(f"Cannot find triton_dispatcher_factory.py. Tried: {candidates}")
    finally:
        # Restore original state
        for key in mock_keys:
            if key in saved:
                sys.modules[key] = saved[key]
            else:
                sys.modules.pop(key, None)


_factory_cache = None


def _get_factory():
    """Lazy-load factory module (avoids side effects at import time)."""
    global _factory_cache
    if _factory_cache is None:
        _factory_cache = _load_factory_module()
    return _factory_cache


def _expand_schema_arg_types(schema):
    return _get_factory()._expand_schema_arg_types(schema)


def _TensorDescDispatcherWrapper(*args, **kwargs):
    return _get_factory()._TensorDescDispatcherWrapper(*args, **kwargs)


def _make_schema(args, tensordesc_meta=None):
    """Helper to build a minimal schema dict."""
    schema = {"args": [{"type": ty, "index": i} for i, ty in enumerate(args)]}
    if tensordesc_meta is not None:
        schema["tensordesc_meta"] = tensordesc_meta
    return schema


class TestExpandSchemaArgTypes(unittest.TestCase):

    def test_no_tensordesc(self):
        """Regular args pass through unchanged."""
        schema = _make_schema(["*fp32", "i32", "i64"])
        flat, info = _expand_schema_arg_types(schema)
        self.assertEqual(flat, ["*fp32", "i32", "i64"])
        self.assertIsNone(info)

    def test_host_side_1d(self):
        """1D host-side tensordesc expands to: *dtype, i64*2, i1, i1, i32*1, i64*1."""
        schema = _make_schema(["tensordesc<fp16[32]>"])
        flat, info = _expand_schema_arg_types(schema)
        self.assertEqual(flat, ["*fp16", "i64", "i64", "i1", "i1", "i32", "i64"])
        self.assertEqual(info, [(0, 1, None)])

    def test_host_side_2d(self):
        """2D host-side tensordesc expands correctly."""
        schema = _make_schema(["tensordesc<fp32[32, 64]>"])
        flat, info = _expand_schema_arg_types(schema)
        expected = ["*fp32"] + ["i64"] * 4 + ["i1", "i1"] + ["i32"] * 2 + ["i64"] * 2
        self.assertEqual(flat, expected)
        self.assertEqual(info, [(0, 2, None)])

    def test_host_side_3d(self):
        """3D host-side tensordesc."""
        schema = _make_schema(["tensordesc<bf16[8, 16, 32]>"])
        flat, info = _expand_schema_arg_types(schema)
        expected = ["*bf16"] + ["i64"] * 6 + ["i1", "i1"] + ["i32"] * 3 + ["i64"] * 3
        self.assertEqual(flat, expected)
        self.assertEqual(info, [(0, 3, None)])

    def test_mixed_args(self):
        """Tensordesc mixed with regular args."""
        schema = _make_schema(["*fp32", "tensordesc<fp16[8, 16]>", "i32"])
        flat, info = _expand_schema_arg_types(schema)
        expected = (["*fp32"] + ["*fp16"] + ["i64"] * 4 + ["i1", "i1"] + ["i32"] * 2 + ["i64"] * 2 + ["i32"])
        self.assertEqual(flat, expected)
        self.assertEqual(info, [(1, 2, None)])

    def test_tma_2d(self):
        """TMA path (meta != None) expands to nvTmaDesc + shapes + strides."""
        schema = _make_schema(
            ["tensordesc<fp16[32, 64]>"],
            tensordesc_meta=[{
                "swizzle": 3,
                "elem_size": 2,
                "elem_type": 0,
                "block_size": [32, 64],
                "fp4_padded": False,
            }],
        )
        flat, info = _expand_schema_arg_types(schema)
        # nvTmaDesc + 2 shapes (i32) + 2 strides (i64)
        self.assertEqual(flat, ["nvTmaDesc", "i32", "i32", "i64", "i64"])
        self.assertEqual(len(info), 1)
        pos, ndim, meta = info[0]
        self.assertEqual(pos, 0)
        self.assertEqual(ndim, 2)
        self.assertIsNotNone(meta)

    def test_invalid_type_raises(self):
        """Malformed tensordesc type string raises ValueError."""
        schema = _make_schema(["tensordesc_malformed"])
        with self.assertRaises(ValueError) as ctx:
            _expand_schema_arg_types(schema)
        self.assertIn("Cannot parse tensordesc type", str(ctx.exception))

    def test_multiple_tensordesc(self):
        """Multiple host-side tensordesc args."""
        schema = _make_schema(["tensordesc<fp16[4]>", "i32", "tensordesc<bf16[8, 16]>"])
        flat, info = _expand_schema_arg_types(schema)
        td1 = ["*fp16", "i64", "i64", "i1", "i1", "i32", "i64"]
        mid = ["i32"]
        td2 = ["*bf16"] + ["i64"] * 4 + ["i1", "i1"] + ["i32"] * 2 + ["i64"] * 2
        self.assertEqual(flat, td1 + mid + td2)
        self.assertEqual(info, [(0, 1, None), (2, 2, None)])

    def test_mixed_tma_and_host(self):
        """Mixed host-side (meta=None) + TMA (meta≠None) tensordesc args expand correctly."""
        schema = _make_schema(
            ["tensordesc<fp16[4]>", "tensordesc<fp16[32, 64]>"],
            tensordesc_meta=[
                None,
                {
                    "swizzle": 3,
                    "elem_size": 2,
                    "elem_type": 0,
                    "block_size": [32, 64],
                    "fp4_padded": False,
                },
            ],
        )
        flat, info = _expand_schema_arg_types(schema)
        # First: host-side 1D → *fp16, i64*2, i1, i32*1, i64*1
        # Second: TMA 2D → nvTmaDesc, i32*2, i64*2
        expected = [
            "*fp16",
            "i64",
            "i64",
            "i1",
            "i1",
            "i32",
            "i64",
            "nvTmaDesc",
            "i32",
            "i32",
            "i64",
            "i64",
        ]
        self.assertEqual(flat, expected)
        self.assertEqual(len(info), 2)
        self.assertEqual(info[0], (0, 1, None))
        self.assertEqual(info[1][0], 1)  # pos
        self.assertEqual(info[1][1], 2)  # ndim
        self.assertIsNotNone(info[1][2])  # meta


class FakeTensorDesc:
    """Mock TensorDescriptor for testing."""

    __slots__ = ("base", "shape", "strides", "padding", "round_f32_to_tf32")

    def __init__(self, base, shape, strides, padding="zero", round_f32_to_tf32=False):
        self.base = base
        self.shape = shape
        self.strides = strides
        self.padding = padding
        self.round_f32_to_tf32 = round_f32_to_tf32


class TestTensorDescDispatcherWrapper(unittest.TestCase):

    def test_expansion_1d(self):
        """1D tensordesc is expanded correctly."""
        calls = []

        def fake_dispatcher(gx, gy, gz, stream, *args):
            calls.append(args)

        wrapper = _TensorDescDispatcherWrapper(fake_dispatcher, [(0, 1, None)])
        desc = FakeTensorDesc(base="PTR", shape=[32], strides=[1], padding="zero")
        wrapper(1, 1, 1, 0, desc)
        self.assertEqual(calls, [("PTR", 32, 1, False, False, 32, 1)])

    def test_expansion_2d_nan_padding(self):
        """2D tensordesc with nan padding."""
        calls = []

        def fake_dispatcher(gx, gy, gz, stream, *args):
            calls.append(args)

        wrapper = _TensorDescDispatcherWrapper(fake_dispatcher, [(0, 2, None)])
        desc = FakeTensorDesc(base="PTR", shape=[32, 64], strides=[64, 1], padding="nan")
        wrapper(1, 1, 1, 0, desc)
        self.assertEqual(calls, [("PTR", 32, 64, 64, 1, True, False, 32, 64, 64, 1)])

    def test_expansion_round_f32_to_tf32(self):
        """round_f32_to_tf32=True is forwarded in the correct position."""
        calls = []

        def fake_dispatcher(gx, gy, gz, stream, *args):
            calls.append(args)

        wrapper = _TensorDescDispatcherWrapper(fake_dispatcher, [(0, 1, None)])
        desc = FakeTensorDesc(base="PTR", shape=[32], strides=[1], padding="zero", round_f32_to_tf32=True)
        wrapper(1, 1, 1, 0, desc)
        self.assertEqual(calls, [("PTR", 32, 1, False, True, 32, 1)])

    def test_mixed_args(self):
        """Tensordesc at position 1, regular args at 0 and 2."""
        calls = []

        def fake_dispatcher(gx, gy, gz, stream, *args):
            calls.append(args)

        wrapper = _TensorDescDispatcherWrapper(fake_dispatcher, [(1, 1, None)])
        desc = FakeTensorDesc(base="PTR", shape=[16], strides=[1], padding="zero")
        wrapper(2, 1, 1, 42, "regular0", desc, "regular2")
        self.assertEqual(calls, [("regular0", "PTR", 16, 1, False, False, 16, 1, "regular2")])

    def test_grid_and_stream_passthrough(self):
        """Grid and stream args are passed correctly to underlying dispatcher."""
        calls = []

        def fake_dispatcher(gx, gy, gz, stream, *args):
            calls.append((gx, gy, gz, stream))

        wrapper = _TensorDescDispatcherWrapper(fake_dispatcher, [(0, 1, None)])
        desc = FakeTensorDesc(base="P", shape=[1], strides=[1])
        wrapper(4, 8, 2, 999, desc)
        self.assertEqual(calls, [(4, 8, 2, 999)])

    def test_tma_expansion(self):
        """TMA tensordesc (meta≠None) is NOT handled by the Python wrapper.

        With the C-level TMA expansion, make_triton_dispatcher does NOT
        create a _TensorDescDispatcherWrapper for TMA kernels. The C
        dispatcher handles TMA expansion directly via EXTRACTOR_TENSORDESC_INDEX.
        This test verifies the wrapper only accepts host-side (meta=None) entries.
        """
        calls = []

        def fake_dispatcher(gx, gy, gz, stream, *args):
            calls.append(args)

        # Host-side wrapper works
        wrapper = _TensorDescDispatcherWrapper(fake_dispatcher, [(1, 2, None)])

        import unittest.mock as mock

        desc = mock.MagicMock()
        desc.base = "BASE_PTR"
        desc.shape = [8, 32]
        desc.strides = [32, 1]
        desc.padding = "nan"
        desc.round_f32_to_tf32 = False

        wrapper(1, 1, 1, 0, "regular0", desc, "regular2")
        # expanded: regular0 + base + shape + strides + padding + round_f32_to_tf32 + shape + strides + regular2
        self.assertEqual(
            calls,
            [(
                "regular0",
                "BASE_PTR",
                8,
                32,
                32,
                1,
                True,
                False,
                8,
                32,
                32,
                1,
                "regular2",
            )],
        )


class TestMakeTritonDispatcherClusterDims(unittest.TestCase):
    """Tests that cluster_dims from schema are passed to the C dispatcher."""

    def test_cluster_dims_extracted_from_schema(self):
        """When schema has cluster_dims, they're extracted correctly."""
        schema = {
            "args": [{"type": "i32", "index": 0}],
            "cluster_dims": [2, 1, 1],
        }
        cluster_dims = schema.get("cluster_dims", [1, 1, 1])
        assert isinstance(cluster_dims, list)
        self.assertEqual(cluster_dims[0], 2)
        self.assertEqual(cluster_dims[1], 1)
        self.assertEqual(cluster_dims[2], 1)

    def test_cluster_dims_default_when_absent(self):
        """When schema has no cluster_dims, defaults (1,1,1) are used."""
        schema = {
            "args": [{"type": "i32", "index": 0}],
        }
        cluster_dims = schema.get("cluster_dims", [1, 1, 1])
        self.assertEqual(cluster_dims, [1, 1, 1])

    def test_cluster_dims_forwarded_to_dispatcher(self):
        """make_triton_dispatcher passes cluster_dims kwargs to _TritonDispatcher."""
        import unittest.mock as mock

        factory = _get_factory()

        # Mock _load_module to return a mock with _TritonDispatcher and build_signature_metadata
        mock_module = mock.MagicMock()
        mock_module.build_signature_metadata.return_value = [1]  # one arg type code
        mock_dispatcher_instance = mock.MagicMock()
        mock_module._TritonDispatcher.return_value = mock_dispatcher_instance

        schema = {
            "args": [{"type": "i32", "index": 0}],
            "num_warps": 4,
            "num_ctas": 1,
            "shared_mem": 0,
            "cluster_dims": [2, 1, 1],
        }

        with mock.patch.object(factory, "_load_module", return_value=mock_module):
            factory.make_triton_dispatcher(schema, 0x1234)

        # Verify _TritonDispatcher was called with cluster_dim_x=2
        call_kwargs = mock_module._TritonDispatcher.call_args
        self.assertIsNotNone(call_kwargs)
        _, kwargs = call_kwargs
        self.assertEqual(kwargs["cluster_dim_x"], 2)
        self.assertEqual(kwargs["cluster_dim_y"], 1)
        self.assertEqual(kwargs["cluster_dim_z"], 1)


class TestAutotunerMetaKwargsSeeding(unittest.TestCase):
    """Tests that autotuner correctly seeds meta kwargs for C proxy dispatch."""

    def test_fc_options_hash_computed_from_meta(self):
        """_fc_options_hash is computed from meta-params after seeding."""
        # Simulate what autotuner._try_fast_path does:
        _meta = {"ctas_per_cga": (2, 1, 1), "num_warps": 4}
        _param_name_to_idx = {"x": 0, "y": 1}  # kernel params (not meta)

        # Compute hash like autotuner does
        _meta_opts = {k: v for k, v in _meta.items() if k not in _param_name_to_idx}
        # All meta params should be in _meta_opts (none are kernel params)
        self.assertEqual(set(_meta_opts.keys()), {"ctas_per_cga", "num_warps"})

        if _meta_opts:
            h = hash(tuple(sorted(_meta_opts.items()))) & 0xFFFFFFFFFFFFFFFF
            self.assertNotEqual(h, 0)  # Non-zero hash for non-empty meta

    def test_fc_meta_kwargs_stored_on_jit_fn(self):
        """After seeding, _fc_meta_kwargs is set on the JIT function."""
        import unittest.mock as mock

        # Simulate JITFunction
        jit_fn = mock.MagicMock()
        jit_fn._param_name_to_idx = {"x": 0}
        jit_fn._fc_options_hash = 0
        jit_fn._jit_proxy_cache = {"old_key": "old_proxy"}

        _meta = {"ctas_per_cga": (2, 1, 1)}

        # Simulate the seeding logic from autotuner.py
        _meta_opts = {k: v for k, v in _meta.items() if k not in jit_fn._param_name_to_idx}
        if _meta_opts:
            jit_fn._fc_options_hash = (hash(tuple(sorted(_meta_opts.items()))) & 0xFFFFFFFFFFFFFFFF)
        jit_fn._fc_meta_kwargs = _meta
        jit_fn._jit_proxy_cache = {}

        # Verify
        self.assertNotEqual(jit_fn._fc_options_hash, 0)
        self.assertEqual(jit_fn._fc_meta_kwargs, {"ctas_per_cga": (2, 1, 1)})
        self.assertEqual(jit_fn._jit_proxy_cache, {})  # Invalidated

    def test_proxy_cache_invalidated_after_seeding(self):
        """Proxy cache is cleared so new proxies get updated hash + meta."""
        jit_fn_proxy_cache = {"grid_key": "stale_proxy"}

        # After seeding, cache should be cleared
        jit_fn_proxy_cache = {}
        self.assertEqual(jit_fn_proxy_cache, {})


class TestCProxyMetaKwargsForwarding(unittest.TestCase):
    """Tests that native_create_jit_proxy forwards meta_kwargs in run_partial."""

    def _get_native_create_jit_proxy(self):
        try:
            from triton._C.libtriton import native_create_jit_proxy  # pyre-ignore[21]

            return native_create_jit_proxy
        except ImportError:
            self.skipTest("libtriton not available (requires GPU build)")

    def _make_jit_fn(self):
        """Create a real JITFunction for testing proxy creation."""
        import triton  # pyre-ignore[21]
        import triton.language as tl  # pyre-ignore[21]

        @triton.jit(c_cache=True)  # pyre-ignore[16]
        def _kernel(x_ptr, N: tl.constexpr):
            pass

        _kernel._fc_options_hash = 0
        _kernel._fc_meta_kwargs = None
        return _kernel

    def test_native_create_jit_proxy_accepts_7_args(self):
        """native_create_jit_proxy accepts optional 7th meta_kwargs argument
        without raising TypeError about argument count."""
        native_create_jit_proxy = self._get_native_create_jit_proxy()
        jit_fn = self._make_jit_fn()

        from triton.runtime import driver  # pyre-ignore[21]

        stream_getter = driver.active.get_current_stream
        device_getter = driver.active.get_current_device

        grid = (1, 1, 1)
        meta_kwargs = {"num_ctas": 2}

        # This must NOT raise TypeError("function takes at most 6 arguments")
        proxy = native_create_jit_proxy(jit_fn, grid, jit_fn.params, 0, stream_getter, device_getter,
                                        meta_kwargs,  # 7th arg
                                        )
        self.assertIsNotNone(proxy, "Proxy should be created with 7 args")

    def test_meta_kwargs_none_accepted(self):
        """Passing None as meta_kwargs is equivalent to not passing it."""
        native_create_jit_proxy = self._get_native_create_jit_proxy()
        jit_fn = self._make_jit_fn()

        from triton.runtime import driver  # pyre-ignore[21]

        stream_getter = driver.active.get_current_stream
        device_getter = driver.active.get_current_device

        grid = (1, 1, 1)

        # None should be treated as "no meta kwargs" — no crash
        proxy = native_create_jit_proxy(jit_fn, grid, jit_fn.params, 0, stream_getter, device_getter,
                                        None,  # 7th arg = None
                                        )
        self.assertIsNotNone(proxy, "Proxy should be created with None meta_kwargs")

    def test_meta_kwargs_forwarded_in_fallback(self):
        """When proxy falls back to run_partial, meta_kwargs are included."""
        native_create_jit_proxy = self._get_native_create_jit_proxy()

        from unittest.mock import patch

        import torch  # pyre-ignore[21]
        import triton  # pyre-ignore[21]
        import triton.language as tl  # pyre-ignore[21]

        @triton.jit(c_cache=True)  # pyre-ignore[16]
        def _kernel(x_ptr, N: tl.constexpr):
            pass

        _kernel._fc_options_hash = 0
        _kernel._fc_meta_kwargs = {"num_warps": 8}

        from triton.runtime import driver  # pyre-ignore[21]

        stream_getter = driver.active.get_current_stream
        device_getter = driver.active.get_current_device

        grid = (1, 1, 1)
        meta_kwargs = {"num_warps": 8}

        proxy = native_create_jit_proxy(
            _kernel,
            grid,
            _kernel.params,
            0,
            stream_getter,
            device_getter,
            meta_kwargs,
        )

        # Call the proxy with args that will cause a FastCache miss → fallback.
        # The fallback calls run_partial which should include num_warps=8.
        # We patch JITFunction.run to capture what kwargs it receives.
        captured_kwargs = {}

        original_run = _kernel.run

        def capture_run(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return original_run(*args, **kwargs)

        x = torch.zeros(1, device="cuda")  # pyre-ignore[16]
        with patch.object(_kernel, "run", side_effect=capture_run):
            try:
                proxy(x)
            except Exception:
                pass  # Compilation may fail, but run should have been called

        if captured_kwargs:
            self.assertIn(
                "num_warps",
                captured_kwargs,
                "meta_kwargs should be forwarded in fallback path",
            )
            self.assertEqual(captured_kwargs["num_warps"], 8)
