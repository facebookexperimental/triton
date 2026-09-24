"""TLX AMD tests -- CDNA4 (gfx950)."""

import contextlib
from dataclasses import dataclass, replace
import importlib
import inspect
import os
import re
import unittest
from unittest import mock

import pytest
import torch
from triton._internal_testing import is_hip_cdna4
from triton.language.extra.tlx.tutorials import amd_fa_varlen_bwd
from triton.tlx.ops.kernels.flash_attn import gfx950_bwd as amd_fa_bwd
from triton.tlx.ops.kernels.flash_attn.gfx950_bwd import (
    _select_d64_dispatch,
    fa_backward,
)


def flash_attention_registry_available() -> bool:
    """True when torch exposes the public conditional-provider APIs."""
    try:
        import torch.nn.attention as attention

        return all(
            hasattr(attention, name) for name in (
                "activate_flash_attention_impl",
                "current_flash_attention_impl",
                "list_flash_attention_impls",
                "register_flash_attention_impl",
                "restore_flash_attention_impl",
            )) and hasattr(torch.library, "get_kernel")
    except ImportError:
        return False


@dataclass(frozen=True)
class ReferenceCase:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    o: torch.Tensor
    do: torch.Tensor
    lse: torch.Tensor
    sm_scale: float
    causal: bool
    grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    @property
    def kernel_args(self):
        return (self.q, self.k, self.v, self.o, self.do, self.lse, self.sm_scale, self.causal)


def _make_d64_aten_case(shape, *, causal, seed):
    batch, hq, hkv, sq, skv, head_dim = shape
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)

    def random(tensor_shape):
        return torch.randn(
            tensor_shape,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ).contiguous()

    q = random((batch, hq, sq, head_dim))
    k = random((batch, hkv, skv, head_dim))
    v = random((batch, hkv, skv, head_dim))
    do = random(q.shape)
    sm_scale = head_dim**-0.5
    state = torch.ops.aten._scaled_dot_product_flash_attention.default(
        q,
        k,
        v,
        0.0,
        causal,
        False,
        scale=sm_scale,
    )
    out, lse, cum_q, cum_k, max_q, max_k, rng, unused, _debug = state
    reference = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
        do,
        q,
        k,
        v,
        out,
        lse,
        cum_q,
        cum_k,
        max_q,
        max_k,
        0.0,
        causal,
        rng,
        unused,
        scale=sm_scale,
    )
    return ReferenceCase(
        q,
        k,
        v,
        out.contiguous(),
        do,
        lse.contiguous(),
        sm_scale,
        causal,
        tuple(reference),
    )


def _assert_scratch_free(name, compiled):
    amdgcn = compiled.asm["amdgcn"]
    private_segments = {
        int(value)
        for value in re.findall(r"(?:\.amdhsa_)?private_segment_fixed_size:?\s+(\d+)", amdgcn)
    }
    resources = {
        "n_spills": compiled.n_spills,
        "global_scratch_bytes": compiled.metadata.global_scratch_size,
        "private_segment_bytes": private_segments.pop() if len(private_segments) == 1 else None,
        "scratch_loads": len(re.findall(r"\bscratch_load", amdgcn)),
        "scratch_stores": len(re.findall(r"\bscratch_store", amdgcn)),
    }
    assert resources == {
        "n_spills": 0,
        "global_scratch_bytes": 0,
        "private_segment_bytes": 0,
        "scratch_loads": 0,
        "scratch_stores": 0,
    }, (name, resources)


def _capture_kernel_with_constexprs(monkeypatch, module, name, **constexprs):
    kernel = getattr(module, name)
    launches = []

    class CapturedKernel:

        def __getitem__(self, grid):

            def launch(*args, **kwargs):
                kwargs.update(constexprs)
                compiled = kernel[grid](*args, **kwargs)
                launches.append((kwargs, compiled))
                return compiled

            return launch

    monkeypatch.setattr(module, name, CapturedKernel())
    return launches


def _make_varlen_d128_reference_case(
    q_lengths,
    kv_lengths,
    *,
    q_heads,
    kv_heads,
    seed,
    causal=False,
    strided_v=False,
    sm_scale=None,
):
    assert q_heads > 0 and kv_heads > 0 and q_heads % kv_heads == 0
    if causal:
        assert q_lengths == kv_lengths
    group_size = q_heads // kv_heads
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    total_q = sum(q_lengths)
    total_kv = sum(kv_lengths)
    scale = 128**-0.5 if sm_scale is None else sm_scale
    q = torch.randn((total_q, q_heads, 128), dtype=torch.bfloat16, device="cuda", generator=generator)
    k = torch.randn((total_kv, kv_heads, 128), dtype=torch.bfloat16, device="cuda", generator=generator)
    if strided_v:
        v_storage = torch.randn(
            (total_kv, 3, kv_heads, 128),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        v = v_storage[:, 0]
    else:
        v = torch.randn((total_kv, kv_heads, 128), dtype=torch.bfloat16, device="cuda", generator=generator)
    do = torch.randn((total_q, q_heads, 128), dtype=torch.bfloat16, device="cuda", generator=generator)
    out = torch.empty_like(q)
    lse = torch.empty((q_heads, total_q), dtype=torch.float32, device="cuda")
    expected_dq = torch.empty_like(q)
    expected_dk = torch.zeros_like(k, dtype=torch.float32)
    expected_dv = torch.zeros_like(v, dtype=torch.float32)

    q_begin = 0
    kv_begin = 0
    for q_length, kv_length in zip(q_lengths, kv_lengths, strict=True):
        q_end = q_begin + q_length
        kv_end = kv_begin + kv_length
        for q_head in range(q_heads):
            kv_head = q_head // group_size
            q_tile = q[q_begin:q_end, q_head].float()
            k_tile = k[kv_begin:kv_end, kv_head].float()
            v_tile = v[kv_begin:kv_end, kv_head].float()
            do_tile = do[q_begin:q_end, q_head].float()
            scores = q_tile @ k_tile.mT * scale
            if causal:
                query_positions = torch.arange(q_length, device="cuda")
                key_positions = torch.arange(kv_length, device="cuda")
                scores = scores.masked_fill(key_positions[None, :] > query_positions[:, None], float("-inf"))
            lse_tile = torch.logsumexp(scores, dim=1)
            p = torch.exp(scores - lse_tile[:, None])
            out_tile = (p @ v_tile).to(torch.bfloat16)
            delta = torch.sum(out_tile.float() * do_tile, dim=1)
            dp = do_tile @ v_tile.mT
            ds = (p * (dp - delta[:, None])).to(torch.bfloat16).float()
            out[q_begin:q_end, q_head] = out_tile
            lse[q_head, q_begin:q_end] = lse_tile
            expected_dq[q_begin:q_end, q_head] = (ds @ k_tile * scale).to(torch.bfloat16)
            expected_dk[kv_begin:kv_end, kv_head].add_(ds.mT @ q_tile * scale)
            expected_dv[kv_begin:kv_end, kv_head].add_(p.to(torch.bfloat16).float().mT @ do_tile)
        q_begin = q_end
        kv_begin = kv_end

    cu_q = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    cu_kv = torch.tensor([0, *torch.tensor(kv_lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    return q, k, v, out, do, lse, cu_q, cu_kv, scale, (
        expected_dq,
        expected_dk.to(torch.bfloat16),
        expected_dv.to(torch.bfloat16),
    )


@unittest.skipUnless(
    flash_attention_registry_available(),
    "Need the public PyTorch FlashAttention provider registry",
)
class TestTLXFlashAttentionProvider(unittest.TestCase):

    class _NativeKernel:

        def __init__(self, result):
            self.result = result
            self.calls = []

        def call_boxed(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return self.result

    @staticmethod
    def _wrapper_args(shape=(1, 1, 2, 4)):
        query = torch.randn(shape)
        key = torch.randn(shape)
        value = torch.randn(shape)
        out_storage = torch.randn((*shape[:-1], 2 * shape[-1]))
        grad_storage = torch.randn_like(out_storage)
        lse_storage = torch.randn((*shape[:-1], 2))
        out = out_storage[..., ::2]
        grad_out = grad_storage[..., ::2]
        logsumexp = lse_storage[..., 0]
        return (
            grad_out,
            query,
            key,
            value,
            out,
            logsumexp,
            None,
            None,
            shape[2],
            shape[2],
            0.0,
            False,
            torch.tensor(0, dtype=torch.int64),
            torch.tensor(0, dtype=torch.int64),
        )

    @staticmethod
    def _performance_tensors(q_shape, k_shape):
        tensors = [mock.Mock(shape=q_shape) for _ in range(6)]
        tensors[1].shape = k_shape
        for tensor in tensors:
            tensor.data_ptr.return_value = 0
        return tensors

    @staticmethod
    def _d64_dispatch(q_shape, k_shape, causal):
        return amd_fa_bwd._select_d64_dispatch(
            q_shape,
            k_shape,
            causal,
            arch="gfx950",
            cu_count=256,
            sm_scale=0.125,
            bases_aligned_16=True,
        )

    def test_import_registers_without_activation(self):
        import torch.nn.attention as attention

        active = attention.current_flash_attention_impl()
        provider = importlib.import_module("triton.tlx.pytorch")
        with mock.patch.object(
                attention,
                "register_flash_attention_impl",
                wraps=attention.register_flash_attention_impl,
        ) as register:
            provider = importlib.reload(provider)

        register.assert_called_once_with(
            provider.PROVIDER_NAME,
            register_fn=provider.register_tlx_gfx950_flash_attention_backward,
        )
        self.assertIn(provider.PROVIDER_NAME, attention.list_flash_attention_impls())
        self.assertEqual(attention.current_flash_attention_impl(), active)

    def test_tlx_route_maps_arguments_without_hidden_copies(self):
        from triton.tlx import pytorch as provider

        args = self._wrapper_args()
        expected = tuple(torch.empty(0) for _ in range(3))
        native = self._NativeKernel(result=None)
        with (
                mock.patch.object(provider, "_tlx_support_error", return_value=None),
                mock.patch.object(provider.gfx950_bwd, "fa_backward", return_value=expected) as tlx_backward,
        ):
            actual = provider._tlx_scaled_dot_product_flash_attention_backward(
                native,
                object(),
                *args,
                scale=None,
            )

        self.assertIs(actual, expected)
        self.assertEqual(native.calls, [])
        tlx_backward.assert_called_once()
        call = tlx_backward.call_args.args
        self.assertIs(call[0], args[1])
        self.assertIs(call[1], args[2])
        self.assertIs(call[2], args[3])
        self.assertIs(call[3], args[4])
        self.assertIs(call[4], args[0])
        self.assertIs(call[5], args[5])
        self.assertEqual(call[6], args[1].shape[-1]**-0.5)
        self.assertIs(call[7], args[11])

    def test_support_gate_runs_before_performance_gate(self):
        from triton.tlx import pytorch as provider

        args = self._wrapper_args()
        with (
                mock.patch.object(
                    provider.gfx950_bwd,
                    "fa_backward_support_error",
                    return_value="kernel contract rejected the call",
                ) as support,
                mock.patch.object(provider, "_is_performance_validated") as performance,
        ):
            error = provider._tlx_support_error(*args[:12], scale=0.5)

        self.assertEqual(error, "kernel contract rejected the call")
        support.assert_called_once_with(args[1], args[2], args[3], args[4], args[0], args[5], 0.5, args[11])
        performance.assert_not_called()

    def test_support_gate_rejects_unsupported_dispatch_contracts(self):
        from triton.tlx import pytorch as provider

        cases = []
        args = list(self._wrapper_args())
        args[10] = 0.1
        cases.append(("dropout", args, False, "dropout_p must be zero"))
        args = list(self._wrapper_args())
        args[6] = torch.tensor([0, 2])
        cases.append(("varlen", args, False, "only dense attention is supported"))
        args = list(self._wrapper_args())
        args[1] = args[1].to_sparse()
        cases.append(("layout", args, False, "query must use strided layout"))
        args = list(self._wrapper_args())
        cases.append(("deterministic", args, True, "deterministic algorithms are enabled"))
        args = list(self._wrapper_args())
        args[1] = torch.randn(1, 2, 4)
        cases.append(("rank", args, False, "query and key must be rank-4 BHSD tensors"))
        args = list(self._wrapper_args())
        args[8] += 1
        cases.append(("sequence length", args, False, "max_q and max_k must match the dense sequence lengths"))

        for name, args, deterministic, expected in cases:
            with (
                    self.subTest(case=name),
                    mock.patch.object(torch, "are_deterministic_algorithms_enabled", return_value=deterministic),
                    mock.patch.object(provider.gfx950_bwd, "fa_backward_support_error") as support,
                    mock.patch.object(provider, "_is_performance_validated") as performance,
            ):
                self.assertEqual(provider._tlx_support_error(*args[:12], scale=None), expected)
                support.assert_not_called()
                performance.assert_not_called()

        args = self._wrapper_args()
        with (
                mock.patch.object(provider.gfx950_bwd, "fa_backward_support_error", return_value=None),
                mock.patch.object(provider, "_is_performance_validated", return_value=False),
        ):
            self.assertEqual(
                provider._tlx_support_error(*args[:12], scale=None),
                "shape is supported but not performance-validated for dispatcher use",
            )

    def test_unsupported_route_calls_captured_kernel_once(self):
        from triton.tlx import pytorch as provider

        args = self._wrapper_args()
        expected = tuple(torch.empty(0) for _ in range(3))
        native = self._NativeKernel(expected)
        keyset = object()
        with (
                mock.patch.object(provider, "_tlx_support_error", return_value="unsupported"),
                mock.patch.object(provider.gfx950_bwd, "fa_backward") as tlx_backward,
        ):
            actual = provider._tlx_scaled_dot_product_flash_attention_backward(
                native,
                keyset,
                *args,
                scale=0.25,
            )

        self.assertIs(actual, expected)
        tlx_backward.assert_not_called()
        self.assertEqual(len(native.calls), 1)
        fallback_args, fallback_kwargs = native.calls[0]
        self.assertIs(fallback_args[0], keyset)
        for actual_arg, expected_arg in zip(fallback_args[1:], args, strict=True):
            self.assertIs(actual_arg, expected_arg)
        self.assertEqual(fallback_kwargs, {"scale": 0.25})

    def test_zero_head_dimension_uses_native_fallback(self):
        from triton.tlx import pytorch as provider

        args = self._wrapper_args(shape=(1, 1, 2, 0))
        expected = tuple(torch.empty(0) for _ in range(3))
        native = self._NativeKernel(expected)
        keyset = object()
        with mock.patch.object(provider.gfx950_bwd, "fa_backward") as tlx_backward:
            actual = provider._tlx_scaled_dot_product_flash_attention_backward(
                native,
                keyset,
                *args,
                scale=None,
            )

        self.assertIs(actual, expected)
        tlx_backward.assert_not_called()
        self.assertEqual(len(native.calls), 1)

    def test_registration_captures_kernel_at_each_activation(self):
        from triton.tlx import pytorch as provider

        first = object()
        second = object()
        libraries = [mock.Mock(), mock.Mock()]
        with (
                mock.patch.object(torch.library, "get_kernel", side_effect=(first, second)) as get_kernel,
                mock.patch.object(torch.library, "Library", side_effect=libraries),
        ):
            first_handle = provider.register_tlx_gfx950_flash_attention_backward()
            second_handle = provider.register_tlx_gfx950_flash_attention_backward()

        self.assertEqual(get_kernel.call_count, 2)
        for library, original in zip(libraries, (first, second), strict=True):
            implementation = library.impl.call_args.args[1]
            self.assertIs(implementation.args[0], original)
            self.assertTrue(library.impl.call_args.kwargs["with_keyset"])
        first_handle.remove()
        second_handle.remove()
        self.assertIsNone(first_handle.library)
        self.assertIsNone(second_handle.library)

    def test_only_measured_signatures_are_selected(self):
        from triton.tlx import pytorch as provider

        q_shape = (1, 16, 4096, 64)
        k_shape = (1, 2, 4096, 64)
        tensors = self._performance_tensors(q_shape, k_shape)
        with mock.patch.object(
                provider.gfx950_bwd,
                "_select_d64_dispatch_for_device",
                return_value=self._d64_dispatch(q_shape, k_shape, False),
        ):
            self.assertTrue(provider._is_performance_validated(*tensors, 0.125, False))
            self.assertFalse(provider._is_performance_validated(*tensors, 0.125, True))

        # A family match is not enough: every shape needs its own provider-path
        # measurement and exact dispatch fingerprint.
        tensors = self._performance_tensors((2, 32, 8192, 64), (2, 4, 8192, 64))
        self.assertFalse(provider._is_performance_validated(*tensors, 0.125, False))

        losing_d128 = (
            ((16, 16, 4096, 128), (16, 16, 4096, 128), False),
            ((16, 64, 2048, 128), (16, 8, 2048, 128), True),
        )
        for q_shape, k_shape, causal in losing_d128:
            with self.subTest(q_shape=q_shape, k_shape=k_shape, causal=causal):
                tensors = self._performance_tensors(q_shape, k_shape)
                self.assertFalse(provider._is_performance_validated(*tensors, 128**-0.5, causal))

    def test_expanded_measured_signatures_select_exact_dispatches(self):
        from triton.tlx import pytorch as provider

        d64_cases = (
            ((2, 32, 16384, 64), (2, 32, 16384, 64), False),
            ((2, 32, 16384, 64), (2, 4, 16384, 64), False),
            ((2, 32, 16384, 64), (2, 32, 16384, 64), True),
            ((2, 32, 16384, 64), (2, 4, 16384, 64), True),
            ((4, 48, 4096, 64), (4, 6, 4096, 64), True),
            ((4, 48, 4096, 64), (4, 6, 8192, 64), True),
            ((4, 48, 4096, 64), (4, 6, 12288, 64), True),
            ((4, 48, 4096, 64), (4, 6, 16384, 64), True),
            ((1, 8, 16384, 64), (1, 8, 16384, 64), True),
            ((1, 8, 16640, 64), (1, 8, 16640, 64), True),
            ((1, 4, 32768, 64), (1, 4, 32768, 64), True),
            ((3, 3, 16384, 64), (3, 3, 16384, 64), True),
            ((2, 8, 16384, 64), (2, 8, 16384, 64), True),
        )
        for q_shape, k_shape, causal in d64_cases:
            tensors = self._performance_tensors(q_shape, k_shape)
            with self.subTest(q_shape=q_shape, k_shape=k_shape, causal=causal), mock.patch.object(
                    provider.gfx950_bwd,
                    "_select_d64_dispatch_for_device",
                    return_value=self._d64_dispatch(q_shape, k_shape, causal),
            ):
                self.assertTrue(provider._is_performance_validated(*tensors, 0.125, causal))

        d128_cases = (
            ((16, 16, 1024, 128), (16, 16, 1024, 128), False),
            ((16, 16, 2048, 128), (16, 16, 2048, 128), False),
            ((16, 64, 1024, 128), (16, 8, 1024, 128), True),
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            for q_shape, k_shape, causal in d128_cases:
                with self.subTest(q_shape=q_shape, k_shape=k_shape, causal=causal):
                    tensors = self._performance_tensors(q_shape, k_shape)
                    self.assertTrue(provider._is_performance_validated(*tensors, 128**-0.5, causal))

    def test_d64_signature_requires_measured_dispatch_config(self):
        from triton.tlx import pytorch as provider

        q_shape = (4, 48, 1024, 64)
        k_shape = (4, 6, 1024, 64)
        tensors = self._performance_tensors(q_shape, k_shape)
        measured = self._d64_dispatch(q_shape, k_shape, True)
        with mock.patch.object(
                provider.gfx950_bwd,
                "_select_d64_dispatch_for_device",
                return_value=measured,
        ):
            self.assertTrue(provider._is_performance_validated(*tensors, 0.125, True))

        mutations = (
            {"family": "causal_m192"},
            {"owner_rows": measured.owner_rows * 2},
            {"key_rows": measured.key_rows * 2},
            {"kv_splits": measured.kv_splits * 2},
            {"selected_causal": not measured.selected_causal},
            {"stat_mode": measured.stat_mode + 1},
            {"dq_logical_n": measured.dq_logical_n * 2},
            {"dq_use_xcd": not measured.dq_use_xcd},
            {"dq_launches": ()},
            {"gqa_grid_mode": None},
            {"cyclic_query_split": not measured.cyclic_query_split},
            {"dkdv_lifetime": None},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), mock.patch.object(
                    provider.gfx950_bwd,
                    "_select_d64_dispatch_for_device",
                    return_value=replace(measured, **mutation),
            ):
                self.assertFalse(provider._is_performance_validated(*tensors, 0.125, True))
        launch = measured.dq_launches[0]
        launch_mutations = (
            {"launch_tiles": launch.launch_tiles + 1},
            {"skip_owner_tail": not launch.skip_owner_tail},
            {"owner_pid_base": launch.owner_pid_base + 1},
            {"launch_q_tiles": launch.launch_q_tiles + 1},
            {"owner_fragments": launch.owner_fragments + 1},
            {"grid_owner_m": launch.grid_owner_m + 1},
        )
        for mutation in launch_mutations:
            with self.subTest(launch_mutation=mutation), mock.patch.object(
                    provider.gfx950_bwd,
                    "_select_d64_dispatch_for_device",
                    return_value=replace(measured, dq_launches=(replace(launch, **mutation), )),
            ):
                self.assertFalse(provider._is_performance_validated(*tensors, 0.125, True))
        with (
                mock.patch.object(
                    provider.gfx950_bwd,
                    "_select_d64_dispatch_for_device",
                    return_value=measured,
                ),
                mock.patch.object(provider.gfx950_bwd, "_d64_q3_register_class", return_value=None),
        ):
            self.assertFalse(provider._is_performance_validated(*tensors, 0.125, True))

        with mock.patch.object(
                provider.gfx950_bwd,
                "_select_d64_dispatch_for_device",
                return_value=object(),
        ):
            self.assertFalse(provider._is_performance_validated(*tensors, 0.125, True))

    def test_short_d128_selects_only_measured_dispatches(self):
        from triton.tlx import pytorch as provider

        tensors = self._performance_tensors((16, 27, 200, 128), (16, 27, 200, 128))
        route_options = {
            amd_fa_bwd._D128_EXACT_ENABLE_ENV: "0",
            amd_fa_bwd._D128_PERSISTENT_ENABLE_ENV: "0",
            amd_fa_bwd._D128_PERSISTENT_PIPE_ENABLE_ENV: "0",
            amd_fa_bwd._D128_SINK_INSTS_ENV: "0",
            amd_fa_bwd._D128_REGCLASS_PRIORITY_ENV: "0",
            amd_fa_bwd._D128_REVERSE_LOCAL_ENV: "0",
        }
        for causal in (False, True):
            with self.subTest(causal=causal, route="split"), mock.patch.dict(os.environ, route_options):
                self.assertTrue(provider._is_performance_validated(*tensors, 128**-0.5, causal))
            with self.subTest(causal=causal, route="exact"), mock.patch.dict(
                    os.environ,
                    route_options | {amd_fa_bwd._D128_EXACT_ENABLE_ENV: "1"},
            ):
                self.assertTrue(provider._is_performance_validated(*tensors, 128**-0.5, causal))
            for option in (amd_fa_bwd._D128_PERSISTENT_ENABLE_ENV, amd_fa_bwd._D128_PERSISTENT_PIPE_ENABLE_ENV):
                with self.subTest(causal=causal, route=option), mock.patch.dict(
                        os.environ,
                        route_options | {option: "1"},
                ):
                    self.assertFalse(provider._is_performance_validated(*tensors, 128**-0.5, causal))
            for option in (
                    amd_fa_bwd._D128_SINK_INSTS_ENV,
                    amd_fa_bwd._D128_REGCLASS_PRIORITY_ENV,
                    amd_fa_bwd._D128_REVERSE_LOCAL_ENV,
            ):
                with self.subTest(causal=causal, route="exact", regalloc=option), mock.patch.dict(
                        os.environ,
                        route_options | {
                            amd_fa_bwd._D128_EXACT_ENABLE_ENV: "1",
                            option: "1",
                        },
                ):
                    self.assertFalse(provider._is_performance_validated(*tensors, 128**-0.5, causal))

        with mock.patch.dict(os.environ, route_options):
            measured = amd_fa_bwd._select_d128_dispatch((16, 27, 200, 128), False)
        mutations = (
            {"entry": object()},
            {"block_m": measured.block_m * 2},
            {"block_n": measured.block_n // 2},
            {"num_warps": measured.num_warps // 2},
            {"pipelined": not measured.pipelined},
            {"rectangular": not measured.rectangular},
            {"exact": not measured.exact},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), mock.patch.object(
                    amd_fa_bwd,
                    "_select_d128_dispatch",
                    return_value=replace(measured, **mutation),
            ):
                self.assertFalse(provider._is_performance_validated(*tensors, 128**-0.5, False))

    def test_every_tensor_base_must_be_aligned(self):
        from triton.tlx import pytorch as provider

        names = ("query", "key", "value", "out", "grad_out", "logsumexp")
        tensors = self._performance_tensors((16, 64, 1024, 128), (16, 8, 1024, 128))
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(provider._is_performance_validated(*tensors, 128**-0.5, False))
            for name, tensor in zip(names, tensors, strict=True):
                with self.subTest(tensor=name):
                    tensor.data_ptr.return_value = 2
                    self.assertFalse(provider._is_performance_validated(*tensors, 128**-0.5, False))
                    tensor.data_ptr.return_value = 0

    def test_d128_experimental_regalloc_options_are_not_selected(self):
        from triton.tlx import pytorch as provider

        tensors = self._performance_tensors((16, 64, 1024, 128), (16, 8, 1024, 128))
        options = (
            amd_fa_bwd._D128_SINK_INSTS_ENV,
            amd_fa_bwd._D128_REGCLASS_PRIORITY_ENV,
            amd_fa_bwd._D128_REVERSE_LOCAL_ENV,
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(provider._is_performance_validated(*tensors, 128**-0.5, False))
        for option in options:
            with self.subTest(option=option), mock.patch.dict(os.environ, {option: "1"}, clear=True):
                self.assertFalse(provider._is_performance_validated(*tensors, 128**-0.5, False))

    def test_interleaved_d128_requires_measured_dispatch_config(self):
        from triton.tlx import pytorch as provider

        tensors = self._performance_tensors((16, 64, 1024, 128), (16, 8, 1024, 128))
        with mock.patch.dict(os.environ, {}, clear=True):
            measured = amd_fa_bwd._select_d128_interleaved_dispatch()
            self.assertTrue(provider._is_performance_validated(*tensors, 128**-0.5, False))

        mutations = (
            {"main_entry": object()},
            {"block_m": measured.block_m * 2},
            {"block_n": measured.block_n // 2},
            {"num_warps": measured.num_warps // 2},
            {"num_stages": measured.num_stages + 1},
            {"matrix_instr_nonkdim": measured.matrix_instr_nonkdim * 2},
            {"convert_entry": object()},
            {"convert_block_m": measured.convert_block_m // 2},
            {"convert_num_warps": measured.convert_num_warps // 2},
            {"sink_insts_to_avoid_spills": not measured.sink_insts_to_avoid_spills},
            {"regclass_priority_trumps_globalness": not measured.regclass_priority_trumps_globalness},
            {"reverse_local_assignment": not measured.reverse_local_assignment},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), mock.patch.object(
                    amd_fa_bwd,
                    "_select_d128_interleaved_dispatch",
                    return_value=replace(measured, **mutation),
            ):
                self.assertFalse(provider._is_performance_validated(*tensors, 128**-0.5, False))

    def test_d256_forced_staged_route_is_not_selected(self):
        from triton.tlx import pytorch as provider

        tensors = self._performance_tensors((32, 1, 2600, 256), (32, 1, 2600, 256))
        with mock.patch.dict(os.environ, {"TLX_FA_BWD_FORCE_STAGED": "1"}, clear=True):
            self.assertTrue(provider._is_performance_validated(*tensors, 256**-0.5, False))
            self.assertFalse(provider._is_performance_validated(*tensors, 256**-0.5, True))
        with mock.patch.dict(os.environ, {}, clear=True):
            measured = amd_fa_bwd._select_d256_dispatch(False)
        mutations = (
            {"entry": object()},
            {"num_warps": measured.num_warps * 2},
            {"staged": not measured.staged},
            {"pipelined": not measured.pipelined},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), mock.patch.object(
                    amd_fa_bwd,
                    "_select_d256_dispatch",
                    return_value=replace(measured, **mutation),
            ):
                self.assertFalse(provider._is_performance_validated(*tensors, 256**-0.5, False))


@unittest.skipUnless(is_hip_cdna4(), "Requires gfx950 hardware")
class TestDenseFABackwardSupportGfx950(unittest.TestCase):

    @staticmethod
    def _aligned_support_args():
        shape = (1, 1, 256, 64)
        q = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
        k = torch.empty_like(q)
        v = torch.empty_like(q)
        out = torch.empty_like(q)
        grad_out = torch.empty_like(q)
        lse = torch.empty(shape[:-1], device="cuda", dtype=torch.float32)
        return [q, k, v, out, grad_out, lse]

    def test_support_gate_rejects_nonfinite_scale(self):
        args = self._aligned_support_args()
        self.assertIsNone(amd_fa_bwd.fa_backward_support_error(*args, 0.125, False))
        for scale in (float("inf"), float("nan")):
            with self.subTest(scale=scale):
                self.assertEqual(
                    amd_fa_bwd.fa_backward_support_error(*args, scale, False),
                    "sm_scale must be a finite number",
                )

    def test_support_gate_rejects_missing_backward_state(self):
        args = self._aligned_support_args()
        for index, expected in (
            (3, "o must match q shape and device"),
            (4, "do must match q shape and device"),
            (5, "lse must be FP32 B,H,N on the same device"),
        ):
            missing = list(args)
            missing[index] = None
            with self.subTest(index=index):
                self.assertEqual(
                    amd_fa_bwd.fa_backward_support_error(*missing, 0.125, False),
                    expected,
                )

    def test_d256_scratch_producer_covers_consumer(self):
        shape = (32, 1, 2600, 256)
        q = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
        k = torch.ones_like(q)
        v = torch.zeros_like(q)
        out = torch.zeros_like(q)
        grad_out = torch.zeros_like(q)
        lse = torch.zeros(shape[:-1], device="cuda", dtype=torch.float32)
        scale = shape[-1]**-0.5

        for causal in (False, True):
            with self.subTest(causal=causal):
                delta = torch.empty(shape[:-1], device="cuda", dtype=torch.float32)
                dq = torch.empty_like(q)
                dk = torch.empty_like(k)
                dv = torch.empty_like(v)
                amd_fa_bwd._run_bwd_preprocess(out, grad_out, delta)
                amd_fa_bwd._run_bwd_d256(
                    q,
                    k,
                    v,
                    grad_out,
                    lse,
                    delta,
                    dq,
                    dk,
                    dv,
                    scale,
                    causal,
                    poison_scratch=True,
                )
                self.assertTrue(torch.isfinite(dq).all().item())


@unittest.skipUnless(
    flash_attention_registry_available() and is_hip_cdna4(),
    "Need the public PyTorch registry and AMD MI350X (gfx950)",
)
class TestTLXFlashAttentionProviderGfx950(unittest.TestCase):

    @contextlib.contextmanager
    def _activated(self):
        import torch.nn.attention as attention
        from triton.tlx import pytorch as provider

        previous = attention.current_flash_attention_impl()
        attention.activate_flash_attention_impl(provider.PROVIDER_NAME)
        try:
            yield provider
        finally:
            attention.restore_flash_attention_impl()
            if previous is not None:
                attention.activate_flash_attention_impl(previous)

    @staticmethod
    def _run_sdpa_grads(query, key, value, grad_out, *, causal, enable_gqa, attention_mask=None):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        q, k, v = (tensor.detach().requires_grad_(True) for tensor in (query, key, value))
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                is_causal=causal,
                enable_gqa=enable_gqa,
            )
        return torch.autograd.grad(out, (q, k, v), grad_out)

    def _assert_sdpa_autograd_routes_to_tlx(
        self,
        query_shape,
        key_shape,
        *,
        causal,
        enable_gqa=False,
        attention_mask=None,
        seed,
        max_relative_l2,
    ):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        query = torch.randn(query_shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        key = torch.randn(key_shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        value = torch.randn(key_shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        grad_out = torch.randn(query_shape, device="cuda", dtype=torch.bfloat16, generator=generator)

        expected = self._run_sdpa_grads(
            query,
            key,
            value,
            grad_out,
            causal=causal,
            enable_gqa=enable_gqa,
            attention_mask=attention_mask,
        )
        with self._activated(), mock.patch.object(
                amd_fa_bwd,
                "fa_backward",
                wraps=amd_fa_bwd.fa_backward,
        ) as tlx_backward:
            actual = self._run_sdpa_grads(
                query,
                key,
                value,
                grad_out,
                causal=causal,
                enable_gqa=enable_gqa,
                attention_mask=attention_mask,
            )
        tlx_backward.assert_called_once()
        self.assertTrue(all(tensor.is_contiguous() for tensor in tlx_backward.call_args.args[:6]))
        for result, reference in zip(actual, expected, strict=True):
            relative_l2 = torch.linalg.vector_norm(result.float() - reference.float()) / torch.linalg.vector_norm(
                reference.float())
            self.assertLess(relative_l2.item(), max_relative_l2)

    def _assert_misaligned_query_uses_native(self, query_shape, key_shape, *, causal, enable_gqa, seed):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        query_storage = torch.randn(
            torch.Size(query_shape).numel() + 1,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        query = query_storage[1:].view(query_shape)
        self.assertTrue(query.is_contiguous())
        self.assertNotEqual(query.data_ptr() % 16, 0)
        key = torch.randn(key_shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        value = torch.randn(key_shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        grad_out = torch.randn(query_shape, device="cuda", dtype=torch.bfloat16, generator=generator)

        expected = self._run_sdpa_grads(query, key, value, grad_out, causal=causal, enable_gqa=enable_gqa)
        with self._activated(), mock.patch.object(
                amd_fa_bwd,
                "fa_backward",
                wraps=amd_fa_bwd.fa_backward,
        ) as tlx_backward:
            actual = self._run_sdpa_grads(query, key, value, grad_out, causal=causal, enable_gqa=enable_gqa)
        tlx_backward.assert_not_called()
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, reference)

    def test_sdpa_autograd_routes_d64_to_tlx(self):
        self._assert_sdpa_autograd_routes_to_tlx(
            (1, 24, 4096, 64),
            (1, 24, 4096, 64),
            causal=True,
            seed=3639,
            max_relative_l2=5e-3,
        )

    def test_sdpa_autograd_routes_d64_noncausal_mha_to_tlx(self):
        self._assert_sdpa_autograd_routes_to_tlx(
            (1, 16, 4096, 64),
            (1, 16, 4096, 64),
            causal=False,
            seed=3642,
            max_relative_l2=5e-3,
        )

    def test_sdpa_autograd_routes_expanded_d64_mha_to_tlx(self):
        self._assert_sdpa_autograd_routes_to_tlx(
            (3, 3, 16384, 64),
            (3, 3, 16384, 64),
            causal=True,
            seed=3670,
            max_relative_l2=5e-3,
        )

    def test_sdpa_autograd_routes_expanded_rectangular_d64_gqa_to_tlx(self):
        from torch.nn.attention.bias import causal_lower_right

        self._assert_sdpa_autograd_routes_to_tlx(
            (4, 48, 4096, 64),
            (4, 6, 8192, 64),
            causal=False,
            enable_gqa=True,
            attention_mask=causal_lower_right(4096, 8192),
            seed=3671,
            max_relative_l2=5e-3,
        )

    def test_sdpa_autograd_routes_d128_gqa_to_tlx(self):
        for sequence_length in (1024, 2048, 4096):
            with self.subTest(sequence_length=sequence_length):
                self._assert_sdpa_autograd_routes_to_tlx(
                    (16, 64, sequence_length, 128),
                    (16, 8, sequence_length, 128),
                    causal=False,
                    enable_gqa=True,
                    seed=3640 + sequence_length,
                    max_relative_l2=1e-2,
                )
        self._assert_sdpa_autograd_routes_to_tlx(
            (16, 64, 1024, 128),
            (16, 8, 1024, 128),
            causal=True,
            enable_gqa=True,
            seed=3672,
            max_relative_l2=1e-2,
        )

    def test_sdpa_autograd_routes_expanded_d128_mha_to_tlx(self):
        for sequence_length in (1024, 2048):
            with self.subTest(sequence_length=sequence_length):
                self._assert_sdpa_autograd_routes_to_tlx(
                    (16, 16, sequence_length, 128),
                    (16, 16, sequence_length, 128),
                    causal=False,
                    seed=3673 + sequence_length,
                    max_relative_l2=1e-2,
                )

    def test_sdpa_autograd_routes_short_d128_to_tlx(self):
        route_options = {
            amd_fa_bwd._D128_EXACT_ENABLE_ENV: "0",
            amd_fa_bwd._D128_PERSISTENT_ENABLE_ENV: "0",
            amd_fa_bwd._D128_PERSISTENT_PIPE_ENABLE_ENV: "0",
            amd_fa_bwd._D128_SINK_INSTS_ENV: "0",
            amd_fa_bwd._D128_REGCLASS_PRIORITY_ENV: "0",
            amd_fa_bwd._D128_REVERSE_LOCAL_ENV: "0",
        }
        for exact in (False, True):
            options = route_options | {amd_fa_bwd._D128_EXACT_ENABLE_ENV: str(int(exact))}
            for causal in (False, True):
                with self.subTest(exact=exact, causal=causal), mock.patch.dict(os.environ, options):
                    self._assert_sdpa_autograd_routes_to_tlx(
                        (16, 27, 200, 128),
                        (16, 27, 200, 128),
                        causal=causal,
                        seed=3650 + 2 * int(exact) + int(causal),
                        max_relative_l2=1e-2,
                    )

    def test_sdpa_autograd_routes_d256_to_tlx(self):
        for causal in (False, True):
            with self.subTest(causal=causal):
                self._assert_sdpa_autograd_routes_to_tlx(
                    (32, 1, 2600, 256),
                    (32, 1, 2600, 256),
                    causal=causal,
                    seed=3641 + causal,
                    max_relative_l2=1e-2,
                )

    def test_measured_causal_d64_shape_with_misaligned_base_uses_native(self):
        self._assert_misaligned_query_uses_native(
            (4, 48, 1024, 64),
            (4, 6, 1024, 64),
            causal=True,
            enable_gqa=True,
            seed=3643,
        )

    def test_measured_d128_shape_with_misaligned_base_uses_native(self):
        self._assert_misaligned_query_uses_native(
            (16, 64, 1024, 128),
            (16, 8, 1024, 128),
            causal=False,
            enable_gqa=True,
            seed=3644,
        )

    def test_unprofitable_shape_uses_native_fallback(self):
        shape = (1, 8, 256, 64)
        q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        grad_out = torch.randn_like(q)
        state = torch.ops.aten._scaled_dot_product_flash_attention.default(q, k, v, 0.0, False, False)
        out, lse, cum_q, cum_k, max_q, max_k, seed, offset, _ = state

        def backward():
            return torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
                grad_out,
                q,
                k,
                v,
                out,
                lse,
                cum_q,
                cum_k,
                max_q,
                max_k,
                0.0,
                False,
                seed,
                offset,
            )

        expected = backward()
        with self._activated(), mock.patch.object(
                amd_fa_bwd,
                "fa_backward",
                wraps=amd_fa_bwd.fa_backward,
        ) as tlx_backward:
            actual = backward()
        tlx_backward.assert_not_called()
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, reference)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize(
    ("shape", "causal", "family"),
    [
        pytest.param((1, 16, 2, 4096, 4096, 64), False, "noncausal_fused_n256", id="fused-gqa8"),
        pytest.param((1, 24, 24, 4096, 4096, 64), True, "causal_scheduled_mha", id="causal-mha"),
        pytest.param((4, 48, 6, 1024, 1024, 64), True, "causal_scheduled_gqa8", id="causal-gqa8"),
        pytest.param(
            (4, 48, 6, 1024, 2048, 64),
            True,
            "causal_scheduled_gqa8",
            id="bottom-right-rectangular-gqa8",
        ),
    ],
)
def test_d64_selected_route_correctness_gfx950(shape, causal, family, monkeypatch):
    monkeypatch.delenv("TRITON_DISABLE_POST_MISCHED", raising=False)
    case = _make_d64_aten_case(shape, causal=causal, seed=311)
    properties = torch.cuda.get_device_properties(case.q.device)
    dispatch = _select_d64_dispatch(
        tuple(case.q.shape),
        tuple(case.k.shape),
        causal,
        arch=properties.gcnArchName,
        cu_count=properties.multi_processor_count,
        sm_scale=case.sm_scale,
        bases_aligned_16=True,
    )
    assert dispatch.family == family

    actual = fa_backward(*case.kernel_args)
    for name, result, expected in zip(("dq", "dk", "dv"), actual, case.grads, strict=True):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_d64_causal_gqa8_codegen_is_scratch_free_gfx950(monkeypatch):
    monkeypatch.delenv("TRITON_DISABLE_POST_MISCHED", raising=False)
    shape = (4, 48, 6, 1024, 1024, 64)
    batch, hq, hkv, sq, skv, head_dim = shape
    q = torch.zeros((batch, hq, sq, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros((batch, hkv, skv, head_dim), device="cuda", dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    o = torch.zeros_like(q)
    do = torch.zeros_like(q)
    lse = torch.zeros((batch, hq, sq), device="cuda", dtype=torch.float32)
    kernels = (
        amd_fa_bwd._attn_bwd_dq_d64_causal_gqa8_kernel,
        amd_fa_bwd._attn_bwd_dkdv_d64_causal_gqa8_kernel,
        amd_fa_bwd._attn_bwd_dkdv_d64_causal_gqa8_reduce_kernel,
    )
    for kernel in kernels:
        kernel.device_caches.clear()

    fa_backward(q, k, v, o, do, lse, 0.125, True)
    torch.cuda.synchronize()

    device = torch.cuda.current_device()
    for kernel in kernels:
        compiled_objects = tuple(kernel.device_caches[device][0].values())
        assert compiled_objects, kernel.fn.__name__
        for compiled in compiled_objects:
            _assert_scratch_free(kernel.fn.__name__, compiled)
            assert not re.search(r"\b\w*atomic\w*\b", compiled.asm["amdgcn"])


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize("register_class", ("default", None, "vgpr", "agpr"), ids=("default", "none", "vgpr", "agpr"))
@pytest.mark.parametrize(
    ("shape", "family", "default_class"),
    (
        pytest.param((1, 8, 8, 16384, 16384, 64), "mha", None, id="mha"),
        pytest.param((1, 16, 2, 8192, 8192, 64), "gqa8", "agpr", id="gqa8"),
    ),
)
def test_d64_q3_register_class_tuning_gfx950(monkeypatch, register_class, shape, family, default_class):
    monkeypatch.delenv("TRITON_DISABLE_POST_MISCHED", raising=False)
    options = {} if register_class == "default" else {"Q3_REGISTER_CLASS": register_class}
    expected_class = default_class if register_class == "default" else register_class
    launches = _capture_kernel_with_constexprs(monkeypatch, amd_fa_bwd, f"_attn_bwd_dq_d64_causal_{family}_kernel",
                                               **options)
    case = _make_d64_aten_case(shape, causal=True, seed=3623)

    actual = fa_backward(*case.kernel_args)
    torch.cuda.synchronize()

    for name, result, expected in zip(("dq", "dk", "dv"), actual, case.grads, strict=True):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        assert relative_l2.item() < 5e-3, (name, relative_l2.item())

    assert len(launches) == 1
    kwargs, compiled = launches[0]
    assert kwargs["OWNER_FRAGMENTS"] == 4
    q3_hints = re.findall(
        r'amdg\.register_resident [^\n]*class "(\w+)" groups (\d+) : tensor<64x64xbf16,',
        compiled.asm["ttir"],
    )
    assert q3_hints == ([] if expected_class is None else [(expected_class, "8")])


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_device_compact_schedules():
    q_lengths = [17, 31, 40]
    kv_lengths = [33, 129, 7]
    cu_q = torch.tensor([0, 17, 48, 88], dtype=torch.int32, device="cuda")
    cu_kv = torch.tensor([0, 33, 162, 169], dtype=torch.int32, device="cuda")

    plan = amd_fa_varlen_bwd.prepare_varlen_backward(
        cu_q,
        cu_kv,
        sum(q_lengths),
        sum(kv_lengths),
        max(q_lengths),
        max(kv_lengths),
    )
    amd_fa_varlen_bwd.validate_varlen_backward_plan(plan)
    q_count, full_count, tail_count, wide_count, plan_error, causal_error = plan.task_counts.tolist()
    assert plan_error == 0
    assert causal_error == 1

    def tasks(sequences, starts, count):
        return set(zip(sequences[:count].tolist(), starts[:count].tolist()))

    assert plan.batch == 3
    assert plan.total_q == 88
    assert plan.total_kv == 169
    assert plan.max_q == 40
    assert tasks(plan.q_block_sequence, plan.q_block_start, q_count) == {(sequence, start)
                                                                         for sequence, length in enumerate(q_lengths)
                                                                         for start in range(0, length, 16)}
    assert tasks(
        plan.full_kv_block_sequence,
        plan.full_kv_block_start,
        full_count,
    ) == {(sequence, start)
          for sequence, length in enumerate(kv_lengths)
          for start in range(0, length // 128 * 128, 128)}
    assert tasks(
        plan.tail_kv_block_sequence,
        plan.tail_kv_block_start,
        tail_count,
    ) == {(sequence, length // 128 * 128)
          for sequence, length in enumerate(kv_lengths)
          if length % 128}
    actual_wide = set(
        zip(
            plan.wide_kv_start[:wide_count].tolist(),
            plan.wide_q_start[:wide_count].tolist(),
            plan.wide_dq_start[:wide_count].tolist(),
            plan.wide_q_len[:wide_count].tolist(),
            plan.wide_kv_valid[:wide_count].tolist(),
        ))
    expected_wide = set()
    q_start = 0
    kv_start = 0
    for sequence, (q_len, kv_len) in enumerate(zip(q_lengths, kv_lengths, strict=True)):
        for block_start in range(0, kv_len, 256):
            expected_wide.add((
                kv_start + block_start,
                q_start,
                q_start + sequence * 15,
                q_len,
                min(256, kv_len - block_start),
            ))
        q_start += q_len
        kv_start += kv_len
    assert actual_wide == expected_wide
    assert plan.cu_seqlens_q is cu_q
    assert plan.cu_seqlens_k is cu_kv
    assert plan.dq_full_kv_sequence is None
    assert plan.dq_full_kv_start is None
    assert plan.dq_tail_k96 is False


@pytest.mark.parametrize(
    ("q_lengths", "kv_lengths", "supply_metadata", "expected_capacities"),
    (
        pytest.param([16, 32], [128, 256], False, (3, 3, 0, 2), id="legacy-all-full"),
        pytest.param([1, 17], [1, 127], False, (3, 0, 2, 2), id="legacy-all-tail"),
        pytest.param([16, 16], [128, 128], True, (2, 2, 0, 2), id="metadata-uniform-all-full"),
        pytest.param([16, 16], [64, 96], True, (2, 0, 2, 3), id="metadata-all-tail"),
        pytest.param([17, 31, 40], [33, 129, 7], True, (9, 1, 3, 4), id="metadata-mixed"),
    ),
)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_plan_tightens_provable_schedule_capacities(q_lengths, kv_lengths, supply_metadata,
                                                                expected_capacities):
    cu_q = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    cu_kv = torch.tensor([0, *torch.tensor(kv_lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    metadata = (sum(q_lengths), sum(kv_lengths), max(q_lengths), max(kv_lengths)) if supply_metadata else ()

    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv, *metadata)

    capacities = (
        plan.q_block_sequence.numel(),
        plan.full_kv_block_sequence.numel(),
        plan.tail_kv_block_sequence.numel(),
        plan.wide_kv_start.numel(),
    )
    assert capacities == expected_capacities
    amd_fa_varlen_bwd.validate_varlen_backward_plan(plan)


def _make_seeded_extend_attention_lengths(batch, max_context, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    prefix = torch.randint(1, max_context // 2, (batch, ), generator=generator)
    extend = torch.randint(1, max_context // 2, (batch, ), generator=generator)
    return extend.tolist(), (prefix + extend).tolist()


def test_varlen_d128_backward_api_defaults_to_noncausal():
    causal = inspect.signature(amd_fa_varlen_bwd.fa_varlen_backward).parameters["causal"]

    assert causal.default is False


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_plan_records_whether_offsets_match():
    shared = torch.tensor([0, 17, 48], dtype=torch.int32, device="cuda")
    different = torch.tensor([0, 17, 49], dtype=torch.int32, device="cuda")

    assert amd_fa_varlen_bwd.prepare_varlen_backward(shared, shared.clone()).qk_offsets_equal is True
    assert amd_fa_varlen_bwd.prepare_varlen_backward(shared, different).qk_offsets_equal is False

    device_plan = amd_fa_varlen_bwd.prepare_varlen_backward(
        shared,
        shared.clone(),
        48,
        48,
        31,
        31,
    )
    assert device_plan.qk_offsets_equal is None
    amd_fa_varlen_bwd.validate_varlen_backward_plan(device_plan, causal=True)

    device_mismatch = amd_fa_varlen_bwd.prepare_varlen_backward(
        shared,
        different,
        48,
        49,
        31,
        32,
    )
    assert device_mismatch.qk_offsets_equal is None
    amd_fa_varlen_bwd.validate_varlen_backward_plan(device_mismatch)
    with pytest.raises(ValueError, match="match between Q and KV"):
        amd_fa_varlen_bwd.validate_varlen_backward_plan(device_mismatch, causal=True)


def test_varlen_d128_seeded_extend_attention_lengths_are_reproducible():
    first = _make_seeded_extend_attention_lengths(batch=19, max_context=12331, seed=42)
    second = _make_seeded_extend_attention_lengths(batch=19, max_context=12331, seed=42)
    q_lengths, kv_lengths = first

    assert first == second
    assert len(q_lengths) == len(kv_lengths) == 19
    assert len(set(q_lengths)) > 1
    assert all(q_length > 0 for q_length in q_lengths)
    assert all(q_length < kv_length for q_length, kv_length in zip(q_lengths, kv_lengths, strict=True))


@pytest.mark.parametrize(
    ("max_q", "group_size", "expected"),
    (
        pytest.param(5460, 3, 3, id="long-gqa3"),
        pytest.param(4096, 4, 4, id="long-gqa4"),
        pytest.param(4096, 6, 3, id="long-gqa6"),
        pytest.param(2048, 8, 4, id="long-gqa8"),
        pytest.param(5456, 3, 1, id="short-gqa3"),
        pytest.param(2032, 8, 1, id="short-gqa8"),
        pytest.param(16384, 1, 1, id="mha"),
        pytest.param(16384, 5, 1, id="unsupported-gqa5"),
    ),
)
def test_varlen_d128_kv_split_selection(max_q, group_size, expected):
    assert amd_fa_varlen_bwd._select_varlen_kv_splits(max_q, group_size) == expected


@pytest.mark.parametrize(
    ("group_size", "kv_splits", "expected"),
    (
        pytest.param(3, 3, (32, 256), id="split-gqa3"),
        pytest.param(8, 4, (32, 256), id="split-gqa8"),
        pytest.param(1, 1, (16, 128), id="mha"),
        pytest.param(3, 1, (16, 128), id="unsplit-gqa"),
    ),
)
def test_varlen_d128_kernel_block_selection(group_size, kv_splits, expected):
    assert amd_fa_varlen_bwd._select_varlen_kernel_blocks(group_size, kv_splits) == expected


def test_varlen_d128_kv_partial_workspace_shapes():
    k = torch.empty((257, 2, 128), device="meta", dtype=torch.bfloat16)
    max_size_k = torch.empty((2**20, 1, 128), device="meta", dtype=torch.bfloat16)
    oversized_k = torch.empty((2**20 + 1, 1, 128), device="meta", dtype=torch.bfloat16)

    direct_dk, direct_dv = amd_fa_varlen_bwd._allocate_varlen_dkdv_partials(k, 1)
    split_dk, split_dv = amd_fa_varlen_bwd._allocate_varlen_dkdv_partials(k, 3)
    max_size_dk, max_size_dv = amd_fa_varlen_bwd._allocate_varlen_dkdv_partials(max_size_k, 4)
    oversized_dk, oversized_dv = amd_fa_varlen_bwd._allocate_varlen_dkdv_partials(oversized_k, 4)

    assert direct_dk is direct_dv is None
    assert max_size_dk is not None and max_size_dv is not None
    assert oversized_dk is oversized_dv is None
    assert split_dk.shape == split_dv.shape == (257, 2, 3, 128)
    assert split_dk.dtype is split_dv.dtype is torch.float32
    assert split_dk.device.type == split_dv.device.type == "meta"


@pytest.mark.parametrize(
    ("q_lengths", "kv_lengths", "q_heads", "kv_heads", "qdo_offsets"),
    (
        pytest.param([7, 31, 65], [33, 257, 7], 1, 1, (0, 0), id="mha1-mixed-full-tail"),
        pytest.param([7, 31, 65], [33, 257, 7], 2, 2, (0, 0), id="mha-mixed-full-tail"),
        pytest.param([1, 17], [1, 127], 2, 2, (0, 0), id="mha-all-tail"),
        pytest.param([16, 32], [128, 256], 2, 2, (0, 0), id="mha-all-full"),
        pytest.param([1, 17], [1, 96], 4, 4, (0, 0), id="mha-k96-all-tail"),
        pytest.param([1, 17, 33], [96, 224, 128], 4, 4, (0, 0), id="mha-tail-96"),
        pytest.param([1, 17, 33], [97, 225, 128], 4, 4, (0, 0), id="mha-tail-97"),
        pytest.param([1, 17, 33], [127, 255, 128], 4, 4, (0, 0), id="mha-tail-127"),
        pytest.param([1, 17, 33], [224, 225, 128], 4, 4, (0, 0), id="mha-mixed-96-97"),
        pytest.param([1, 17, 33], [192, 193, 224], 4, 4, (0, 0), id="mha-mixed-64-65-96"),
        pytest.param([1, 16, 17, 511, 512], [1, 128, 129, 257, 256], 4, 4, (0, 0), id="mha-stats-cache-boundary"),
        pytest.param([1, 17, 513, 1025], [1, 129, 257, 256], 2, 2, (0, 0), id="mha-stats-cache-fallback"),
        pytest.param([17], [129], 2, 2, (1, 0), id="mha-unaligned-q"),
        pytest.param([17], [129], 2, 2, (0, 4), id="mha-unaligned-do"),
        pytest.param([7, 31, 65], [33, 257, 7], 6, 2, (0, 0), id="gqa3-mixed"),
        pytest.param([1, 17], [1, 129], 8, 1, (0, 0), id="gqa8-tail"),
        pytest.param([5460], [17], 3, 1, (0, 0), id="gqa3-long-split3"),
        pytest.param([5460], [17], 12, 4, (0, 0), id="gqa3-multi-kv-long-split3"),
        pytest.param([2048], [17], 8, 1, (0, 0), id="gqa8-long-split4"),
        pytest.param(
            *_make_seeded_extend_attention_lengths(batch=5, max_context=96, seed=443),
            12,
            4,
            (0, 0),
            id="gqa3-seeded-prefix-extend",
        ),
    ),
)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_interleaved_lengths_gfx950(q_lengths, kv_lengths, q_heads, kv_heads, qdo_offsets):
    case = _make_varlen_d128_reference_case(
        q_lengths,
        kv_lengths,
        q_heads=q_heads,
        kv_heads=kv_heads,
        seed=431,
    )
    q, k, v, out, do, lse, cu_q, cu_kv, scale, expected = case
    shifted_inputs = []
    for tensor, offset in zip((q, do), qdo_offsets, strict=True):
        if offset:
            storage = torch.empty(tensor.numel() + offset, dtype=tensor.dtype, device=tensor.device)
            shifted = storage[offset:].view_as(tensor)
            shifted.copy_(tensor)
            assert shifted.is_contiguous() and shifted.data_ptr() % 16 != 0
            tensor = shifted
        shifted_inputs.append(tensor)
    q, do = shifted_inputs
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(
        cu_q,
        cu_kv,
        q.shape[0],
        k.shape[0],
        max(q_lengths),
        max(kv_lengths),
    )

    actual = amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)

    for name, result, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - reference.float()) / torch.linalg.vector_norm(
            reference.float())
        assert relative_l2.item() < 1e-2, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_causal_mha_boundaries_gfx950():
    lengths = [1, 15, 16, 17, 127, 128, 129, 255, 256, 257]
    case = _make_varlen_d128_reference_case(
        lengths,
        lengths,
        q_heads=4,
        kv_heads=4,
        seed=541,
        causal=True,
    )
    q, k, v, out, do, lse, cu_q, cu_kv, scale, expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)

    actual = amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale, causal=True)

    for name, result, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - reference.float()) / torch.linalg.vector_norm(
            reference.float())
        assert relative_l2.item() < 1e-2, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_causal_accepts_tritonbench_strided_v_gfx950():
    lengths = [17, 129, 257]
    case = _make_varlen_d128_reference_case(
        lengths,
        lengths,
        q_heads=4,
        kv_heads=4,
        seed=547,
        causal=True,
        strided_v=True,
    )
    q, k, v, out, do, lse, cu_q, cu_kv, scale, expected = case
    assert v.stride() == (3 * 4 * 128, 128, 1)
    assert not v.is_contiguous()
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)

    actual = amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale, causal=True)

    assert actual[2].is_contiguous()
    for name, result, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - reference.float()) / torch.linalg.vector_norm(
            reference.float())
        assert relative_l2.item() < 1e-2, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_causal_strided_v_rebases_high_token_offsets_gfx950():
    prefix_length = 912
    prefix_sequences = 767
    checked_length = 17
    checked_start = prefix_length * prefix_sequences
    total = checked_start + checked_length
    heads = 4
    dim = 128
    scale = dim**-0.5
    assert checked_start * (3 * heads * dim) > 2**30

    q = torch.zeros((total, heads, dim), dtype=torch.bfloat16, device="cuda")
    k = torch.zeros_like(q)
    v_storage = torch.zeros((total, 3, heads, dim), dtype=torch.bfloat16, device="cuda")
    v = v_storage[:, 0]
    out = torch.zeros_like(q)
    do = torch.zeros_like(q)
    lse = torch.empty((heads, total), dtype=torch.float32, device="cuda")
    prefix_lse = torch.arange(1, prefix_length + 1, dtype=torch.float32, device="cuda").log().repeat(prefix_sequences)
    lse[:, :checked_start] = prefix_lse

    generator = torch.Generator(device="cuda").manual_seed(549)
    checked = slice(checked_start, total)
    q[checked] = torch.randn((checked_length, heads, dim), dtype=torch.bfloat16, device="cuda", generator=generator)
    k[checked] = torch.randn((checked_length, heads, dim), dtype=torch.bfloat16, device="cuda", generator=generator)
    v[checked] = torch.randn((checked_length, heads, dim), dtype=torch.bfloat16, device="cuda", generator=generator)
    do[checked] = torch.randn((checked_length, heads, dim), dtype=torch.bfloat16, device="cuda", generator=generator)

    expected_dq = torch.empty((checked_length, heads, dim), dtype=torch.bfloat16, device="cuda")
    expected_dk = torch.empty_like(expected_dq)
    expected_dv = torch.empty_like(expected_dq)
    causal_mask = torch.ones((checked_length, checked_length), dtype=torch.bool, device="cuda").triu(1)
    for head in range(heads):
        q_tile = q[checked, head].float()
        k_tile = k[checked, head].float()
        v_tile = v[checked, head].float()
        do_tile = do[checked, head].float()
        scores = (q_tile @ k_tile.mT * scale).masked_fill(causal_mask, float("-inf"))
        lse_tile = torch.logsumexp(scores, dim=1)
        p = torch.exp(scores - lse_tile[:, None])
        out_tile = (p @ v_tile).to(torch.bfloat16)
        delta = torch.sum(out_tile.float() * do_tile, dim=1)
        ds = (p * (do_tile @ v_tile.mT - delta[:, None])).to(torch.bfloat16).float()
        out[checked, head] = out_tile
        lse[head, checked] = lse_tile
        expected_dq[:, head] = (ds @ k_tile * scale).to(torch.bfloat16)
        expected_dk[:, head] = (ds.mT @ q_tile * scale).to(torch.bfloat16)
        expected_dv[:, head] = (p.to(torch.bfloat16).float().mT @ do_tile).to(torch.bfloat16)

    lengths = [prefix_length] * prefix_sequences + [checked_length]
    cu = torch.tensor([0, *lengths], dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu, cu)

    actual = amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale, causal=True)

    assert v.stride() == (3 * heads * dim, dim, 1)
    assert actual[2].is_contiguous()
    for name, result, reference in zip(
        ("dq", "dk", "dv"),
        (actual[0][checked], actual[1][checked], actual[2][checked]),
        (expected_dq, expected_dk, expected_dv),
            strict=True,
    ):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - reference.float()) / torch.linalg.vector_norm(
            reference.float())
        assert relative_l2.item() < 1e-2, (name, relative_l2.item())


@pytest.mark.parametrize("sm_scale", [0.0, -0.125], ids=["zero-scale", "negative-scale"])
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_causal_masks_after_score_scaling_gfx950(sm_scale):
    lengths = [17, 129]
    case = _make_varlen_d128_reference_case(
        lengths,
        lengths,
        q_heads=4,
        kv_heads=4,
        seed=551,
        causal=True,
        sm_scale=sm_scale,
    )
    q, k, v, out, do, lse, cu_q, cu_kv, scale, expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)

    actual = amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale, causal=True)

    for name, result, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True):
        assert torch.isfinite(result).all(), name
        error = torch.linalg.vector_norm(result.float() - reference.float())
        scale_norm = torch.clamp(torch.linalg.vector_norm(reference.float()), min=1.0)
        assert (error / scale_norm).item() < 1e-2, (name, error.item(), scale_norm.item())


@pytest.mark.parametrize(
    ("q_lengths", "q_heads", "kv_heads"),
    (
        pytest.param([1, 17, 31, 32, 33, 5460], 3, 1, id="gqa3"),
        pytest.param([1, 17, 31, 32, 33, 2048], 8, 1, id="gqa8"),
    ),
)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_bm32_boundaries_gfx950(q_lengths, q_heads, kv_heads):
    kv_lengths = [1, 255, 256, 257, 511, 17]
    case = _make_varlen_d128_reference_case(
        q_lengths,
        kv_lengths,
        q_heads=q_heads,
        kv_heads=kv_heads,
        seed=487 + q_heads,
    )
    q, k, v, out, do, lse, cu_q, cu_kv, scale, expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(
        cu_q,
        cu_kv,
        q.shape[0],
        k.shape[0],
        max(q_lengths),
        max(kv_lengths),
    )

    actual = amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)

    for name, result, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - reference.float()) / torch.linalg.vector_norm(
            reference.float())
        assert relative_l2.item() < 1e-2, (name, relative_l2.item())


@pytest.mark.parametrize("register_class", ("default", None, "vgpr", "agpr"), ids=("default", "none", "vgpr", "agpr"))
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_prefix_register_class_tuning_gfx950(monkeypatch, register_class):
    options = {} if register_class == "default" else {"K_PREFIX_REGISTER_CLASS": register_class}
    expected_class = None if register_class == "default" else register_class
    launches = _capture_kernel_with_constexprs(monkeypatch, amd_fa_varlen_bwd, "_varlen_bwd_interleaved_bm32_kernel",
                                               **options)
    case = _make_varlen_d128_reference_case(
        [1, 17, 31, 32, 33, 5460],
        [1, 255, 256, 257, 511, 17],
        q_heads=3,
        kv_heads=1,
        seed=490,
    )
    q, k, v, out, do, lse, cu_q, cu_kv, scale, expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)

    actual = amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)
    torch.cuda.synchronize()

    for name, result, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - reference.float()) / torch.linalg.vector_norm(
            reference.float())
        assert relative_l2.item() < 1e-2, (name, relative_l2.item())

    assert len(launches) == 1
    kwargs, compiled = launches[0]
    assert kwargs["KV_SPLITS"] == 3
    prefix_hints = re.findall(
        r'amdg\.register_resident [^\n]*class "(\w+)" groups (\d+) : tensor<256x32xbf16,',
        compiled.asm["ttir"],
    )
    assert prefix_hints == ([] if expected_class is None else [(expected_class, "4")])


@pytest.mark.parametrize("metadata", ("legacy", "missing_sequence", "missing_start", "default_k96"))
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_tail_finalizer_plan_defaults_gfx950(metadata):
    case = _make_varlen_d128_reference_case([1, 17, 33], [1, 225, 256], q_heads=4, kv_heads=4, seed=1901)
    q, k, v, out, do, lse, cu_q, cu_kv, scale, expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)
    if metadata == "legacy":
        fields = {
            key: value
            for key, value in vars(plan).items()
            if key not in ("dq_full_kv_sequence", "dq_full_kv_start", "dq_tail_k96")
        }
        plan = amd_fa_varlen_bwd.VarlenBackwardPlan(**fields)
        assert plan.dq_full_kv_sequence is plan.dq_full_kv_start is None
    elif metadata == "missing_sequence":
        plan = replace(plan, dq_full_kv_sequence=None)
    elif metadata == "missing_start":
        plan = replace(plan, dq_full_kv_start=None)
    else:
        plan = amd_fa_varlen_bwd.VarlenBackwardPlan(
            **{key: value
               for key, value in vars(plan).items()
               if key != "dq_tail_k96"})
    assert plan.dq_tail_k96 is False

    actual = amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)
    for name, result, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True):
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - reference.float()) / torch.linalg.vector_norm(
            reference.float())
        assert relative_l2.item() < 1e-2, (name, relative_l2.item())


@pytest.mark.parametrize("kv_length", (128, 224))
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_runtime_totals_reuse_specialization_gfx950(kv_length):
    kernels = (
        amd_fa_varlen_bwd._varlen_bwd_interleaved_kernel,
        amd_fa_varlen_bwd._varlen_mha_dq_convert_coalesced_kernel,
    )
    for kernel in kernels:
        kernel.device_caches.clear()

    for q_length in (17, 33):
        case = _make_varlen_d128_reference_case([q_length], [kv_length], q_heads=1, kv_heads=1, seed=419 + q_length)
        q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
        plan = amd_fa_varlen_bwd.prepare_varlen_backward(
            cu_q,
            cu_kv,
            q.shape[0],
            k.shape[0],
            q_length,
            kv_length,
        )

        amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)
        torch.cuda.synchronize()

    device = torch.cuda.current_device()
    # Uniform full-tile metadata proves the tail schedule is empty, so only
    # the full specialization is compiled for KV128.
    expected_counts = (1, 1) if kv_length == 128 else (2, 1)
    for kernel, expected in zip(kernels, expected_counts, strict=True):
        assert len(kernel.device_caches[device][0]) == expected, kernel.fn.__name__


@pytest.mark.parametrize(
    ("q_heads", "kv_heads", "long_q"),
    (
        pytest.param(1, 1, None, id="1"),
        pytest.param(4, 4, None, id="4"),
        pytest.param(3, 1, 5460, id="gqa3"),
        pytest.param(8, 1, 2048, id="gqa8"),
    ),
)
@pytest.mark.parametrize("tail_bound", (96, 127))
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_scratch_reset_graph_replay_gfx950(monkeypatch, q_heads, kv_heads, long_q, tail_bound):
    q_lengths = [1, 15, 16, 17, 31, 32, 33, 63, 64, 65]
    kv_lengths = [1, tail_bound, 128, 129, 128 + tail_bound, 256, 257, 33, 7, 384 + tail_bound]
    if long_q is not None:
        # Trigger split GQA while keeping the dense reference small.
        q_lengths.append(long_q)
        kv_lengths.append(17)
    case = _make_varlen_d128_reference_case(
        q_lengths,
        kv_lengths,
        q_heads=q_heads,
        kv_heads=kv_heads,
        seed=557 + q_heads,
    )
    q, k, v, out, do, lse, cu_q, cu_kv, scale, expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)
    preprocess = amd_fa_varlen_bwd._varlen_bwd_preprocess

    class PoisonedPreprocess:

        def __getitem__(self, grid):

            def launch(o, do, delta, cu_q, dq_acc, total_q_padded, task_counts, **kwargs):
                # Valid dQ columns use the whole swizzled BM16 footprint,
                # including rows beyond the final logical query row.
                assert kwargs["ZERO_DQ"]
                dq_acc.fill_(float("nan"))
                return preprocess[grid](o, do, delta, cu_q, dq_acc, total_q_padded, task_counts, **kwargs)

            return launch

    monkeypatch.setattr(amd_fa_varlen_bwd, "_varlen_bwd_preprocess", PoisonedPreprocess())

    def backward():
        return amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)

    def check(actual):
        for name, result, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True):
            assert torch.isfinite(result).all(), name
            relative_l2 = torch.linalg.vector_norm(result.float() - reference.float()) / torch.linalg.vector_norm(
                reference.float())
            assert relative_l2.item() < 1e-2, (name, relative_l2.item())

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            actual = backward()
    torch.cuda.current_stream().wait_stream(stream)
    check(actual)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = backward()
    for _ in range(3):
        for result in actual:
            result.fill_(float("nan"))
        graph.replay()
        check(actual)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_bm32_runtime_totals_reuse_specialization_gfx950():
    kernel = getattr(amd_fa_varlen_bwd, "_varlen_bwd_interleaved_bm32_kernel", None)
    assert kernel is not None
    kernel.device_caches.clear()

    # Keep Triton's ordinary integer divisibility specialization identical;
    # only the packed token count should differ.
    for q_length in (5460, 5476):
        case = _make_varlen_d128_reference_case([q_length], [17], q_heads=3, kv_heads=1, seed=503 + q_length)
        q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
        plan = amd_fa_varlen_bwd.prepare_varlen_backward(
            cu_q,
            cu_kv,
            q.shape[0],
            k.shape[0],
            q_length,
            17,
        )
        amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)
        torch.cuda.synchronize()

    device = torch.cuda.current_device()
    assert len(kernel.device_caches[device][0]) == 1


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_plan_rejects_invalid_offset_metadata():
    valid = torch.tensor([0, 16, 48], dtype=torch.int32, device="cuda")
    strided = torch.tensor(
        [0, -1, 16, -1, 48],
        dtype=torch.int32,
        device="cuda",
    )[::2]
    cases = (
        (valid.view(1, 3), valid, "rank-1 tensor"),
        (valid.to(torch.int64), valid, "dtype torch.int32"),
        (valid, torch.tensor([0, 32], dtype=torch.int32, device="cuda"), "same batch"),
        (
            strided,
            valid,
            "must be contiguous when token metadata is supplied",
        ),
    )
    for cu_q, cu_kv, message in cases:
        with pytest.raises(ValueError, match=message):
            amd_fa_varlen_bwd.prepare_varlen_backward(
                cu_q,
                cu_kv,
                48,
                48,
                32,
                32,
            )


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_legacy_plan_accepts_strided_offsets():
    cu_q = torch.tensor(
        [0, -1, 17, -1, 48],
        dtype=torch.int32,
        device="cuda",
    )[::2]
    cu_kv = torch.tensor(
        [0, -1, 33, -1, 162],
        dtype=torch.int32,
        device="cuda",
    )[::2]

    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)

    assert plan.cu_seqlens_q.is_contiguous()
    assert plan.cu_seqlens_k.is_contiguous()
    assert plan.cu_seqlens_q.tolist() == [0, 17, 48]
    assert plan.cu_seqlens_k.tolist() == [0, 33, 162]

    cu_q.zero_()
    cu_kv.zero_()
    assert plan.cu_seqlens_q.tolist() == [0, 17, 48]
    assert plan.cu_seqlens_k.tolist() == [0, 33, 162]
    amd_fa_varlen_bwd.validate_varlen_backward_plan(plan)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize(
    ("q_offsets", "kv_offsets", "total_q", "total_kv", "max_q", "max_kv"),
    (
        pytest.param([1, 17, 49], [0, 16, 48], 48, 48, 32, 32, id="nonzero-start"),
        pytest.param([0, 16, 16], [0, 16, 48], 16, 48, 16, 32, id="empty-sequence"),
        pytest.param([0, 17, 16], [0, 16, 48], 16, 48, 17, 32, id="nonmonotonic"),
        pytest.param([0, 16, 48], [0, 16, 48], 47, 48, 32, 32, id="wrong-q-total"),
        pytest.param([0, 16], [0, 256], 16, 128, 16, 256, id="undersized-kv-total"),
        pytest.param(
            [0, 16, 32, 48, 64],
            [0, 128, 0, 128, 0],
            64,
            128,
            16,
            128,
            id="alternating-kv-offsets",
        ),
        pytest.param([0, 16, 48], [0, 16, 48], 48, 48, 31, 32, id="wrong-max"),
    ),
)
def test_varlen_d128_device_validation_rejects_invalid_values(q_offsets, kv_offsets, total_q, total_kv, max_q, max_kv):
    cu_q = torch.tensor(q_offsets, dtype=torch.int32, device="cuda")
    cu_kv = torch.tensor(kv_offsets, dtype=torch.int32, device="cuda")

    plan = amd_fa_varlen_bwd.prepare_varlen_backward(
        cu_q,
        cu_kv,
        total_q,
        total_kv,
        max_q,
        max_kv,
    )

    counts = plan.task_counts.tolist()
    assert counts[-2:] == [1, 1]
    assert counts[:-2] == [0, 0, 0, 0]
    with pytest.raises(ValueError, match="cu_seqlens must start at zero"):
        amd_fa_varlen_bwd.validate_varlen_backward_plan(plan)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_backward_rejects_unsupported_signature():
    case = _make_varlen_d128_reference_case([16], [128], q_heads=1, kv_heads=1, seed=407)
    q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv, q.shape[0], k.shape[0], 16, 128)

    with pytest.raises(ValueError, match="q must be contiguous bfloat16 THD"):
        amd_fa_varlen_bwd.fa_varlen_backward(q.float(), k, v, out, do, lse, plan, scale)
    with pytest.raises(ValueError, match="head dimension 128"):
        amd_fa_varlen_bwd.fa_varlen_backward(
            q[..., :64],
            k[..., :64],
            v[..., :64],
            out[..., :64],
            do[..., :64],
            lse,
            plan,
            scale,
        )
    with pytest.raises(ValueError, match="positive Q and KV head counts"):
        amd_fa_varlen_bwd.fa_varlen_backward(
            q[:, :0],
            k[:, :0],
            v[:, :0],
            out[:, :0],
            do[:, :0],
            lse[:0],
            plan,
            scale,
        )
    with pytest.raises(ValueError, match="positive Q and KV head counts"):
        amd_fa_varlen_bwd.fa_varlen_backward(
            q,
            k[:, :0],
            v[:, :0],
            out,
            do,
            lse,
            plan,
            scale,
        )
    with pytest.raises(ValueError, match="sm_scale must be finite"):
        amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, float("nan"))


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_backward_rejects_nondivisible_gqa():
    case = _make_varlen_d128_reference_case([16], [128], q_heads=4, kv_heads=2, seed=411)
    q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv, q.shape[0], k.shape[0], 16, 128)
    invalid_k = torch.empty((k.shape[0], 3, 128), dtype=k.dtype, device=k.device)
    invalid_v = torch.empty_like(invalid_k)

    with pytest.raises(ValueError, match="divisible"):
        amd_fa_varlen_bwd.fa_varlen_backward(q, invalid_k, invalid_v, out, do, lse, plan, scale)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_causal_rejects_cross_attention_offsets():
    case = _make_varlen_d128_reference_case([16], [128], q_heads=1, kv_heads=1, seed=523)
    q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)

    with pytest.raises(ValueError, match="identical Q and KV cumulative offsets"):
        amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale, causal=True)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_causal_rejects_gqa():
    case = _make_varlen_d128_reference_case([17], [17], q_heads=2, kv_heads=1, seed=527)
    q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)

    with pytest.raises(ValueError, match="equal Q and KV head counts"):
        amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale, causal=True)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_causal_rejects_v_with_nondense_head_axes():
    case = _make_varlen_d128_reference_case([17], [17], q_heads=2, kv_heads=2, seed=531, causal=True)
    q, k, _v, out, do, lse, cu_q, cu_kv, scale, _expected = case
    storage = torch.empty((17, 2, 128, 2), dtype=torch.bfloat16, device="cuda")
    v = storage[..., 0]
    assert v.shape == k.shape
    assert v.stride(-1) == 2
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)

    with pytest.raises(ValueError, match="dense head/D axes"):
        amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale, causal=True)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_noncausal_still_rejects_strided_v():
    case = _make_varlen_d128_reference_case(
        [17],
        [17],
        q_heads=2,
        kv_heads=2,
        seed=533,
        strided_v=True,
    )
    q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)

    with pytest.raises(ValueError, match="v must be contiguous bfloat16 THD"):
        amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_backward_rejects_noncontiguous_lse():
    case = _make_varlen_d128_reference_case([16], [128], q_heads=2, kv_heads=2, seed=413)
    q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv, q.shape[0], k.shape[0], 16, 128)
    lse_storage = torch.empty((2, 32), dtype=torch.float32, device="cuda")
    strided_lse = lse_storage[:, ::2]
    assert strided_lse.shape == lse.shape
    assert not strided_lse.is_contiguous()

    with pytest.raises(ValueError, match="lse must be contiguous FP32"):
        amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, strided_lse, plan, scale)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_interleaved_codegen_is_scratch_free_gfx950():
    kernels = (
        amd_fa_varlen_bwd._varlen_bwd_preprocess,
        amd_fa_varlen_bwd._varlen_bwd_interleaved_kernel,
        amd_fa_varlen_bwd._varlen_dq_convert_kernel,
    )
    for kernel in kernels:
        kernel.device_caches.clear()

    case = _make_varlen_d128_reference_case([17], [129], q_heads=3, kv_heads=1, seed=409)
    q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv, q.shape[0], k.shape[0], 17, 129)
    amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)
    torch.cuda.synchronize()

    device = torch.cuda.current_device()
    expected_shared = (256, 64_640, 0)
    expected_specializations = (1, 2, 1)
    for kernel, shared, specialization_count in zip(kernels, expected_shared, expected_specializations, strict=True):
        compiled_objects = tuple(kernel.device_caches[device][0].values())
        assert len(compiled_objects) == specialization_count
        for compiled in compiled_objects:
            _assert_scratch_free(kernel.fn.__name__, compiled)
            assert compiled.metadata.num_warps == 4
            assert compiled.metadata.shared == shared

    for interleaved in kernels[1].device_caches[device][0].values():
        ttir = interleaved.asm["ttir"]
        assert "amdg.rematerialized_range 0 to 128 identity 32" not in ttir
        assert "amdg.rematerialized_range 0 to 16 identity 33" not in ttir
        assert "arith.cmpi sle" not in ttir
        assert "buffer_atomic_pk_add_bf16" in interleaved.asm["amdgcn"]


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_causal_codegen_is_scratch_free_gfx950():
    kernel = amd_fa_varlen_bwd._varlen_bwd_interleaved_kernel
    kernel.device_caches.clear()

    case = _make_varlen_d128_reference_case(
        [129],
        [129],
        q_heads=4,
        kv_heads=4,
        seed=557,
        causal=True,
        strided_v=True,
    )
    q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv)
    amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale, causal=True)
    torch.cuda.synchronize()

    device = torch.cuda.current_device()
    compiled_objects = tuple(kernel.device_caches[device][0].values())
    assert len(compiled_objects) == 2
    for compiled in compiled_objects:
        _assert_scratch_free(kernel.fn.__name__, compiled)
        assert compiled.metadata.num_warps == 4
        assert compiled.metadata.shared == 64_640
        ttir = compiled.asm["ttir"]
        assert ttir.count("amdg.rematerialized_range 0 to 128 identity 32") == 1
        assert ttir.count("amdg.rematerialized_range 0 to 16 identity 33") == 1
        assert ttir.count("arith.cmpi sle") == 1
        assert "arith.select" in ttir
        assert re.search(r"tt\.addptr %V, %\w+ : !tt\.ptr<bf16>, i64", ttir)
        assert "buffer_atomic_pk_add_bf16" in compiled.asm["amdgcn"]


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_split_codegen_is_scratch_free_gfx950():
    kernels = (
        amd_fa_varlen_bwd._varlen_bwd_interleaved_bm32_kernel,
        amd_fa_varlen_bwd._varlen_dkdv_reduce_kernel,
    )
    for kernel in kernels:
        kernel.device_caches.clear()

    for q_length, q_heads in ((5460, 3), (2048, 8)):
        case = _make_varlen_d128_reference_case([q_length], [129], q_heads=q_heads, kv_heads=1, seed=463 + q_heads)
        q, k, v, out, do, lse, cu_q, cu_kv, scale, _expected = case
        plan = amd_fa_varlen_bwd.prepare_varlen_backward(
            cu_q,
            cu_kv,
            q.shape[0],
            k.shape[0],
            q_length,
            129,
        )
        amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale)
    torch.cuda.synchronize()

    device = torch.cuda.current_device()
    for kernel, expected_shared, specialization_count in zip(kernels, (118_048, 0), (2, 2), strict=True):
        compiled_objects = tuple(kernel.device_caches[device][0].values())
        assert len(compiled_objects) == specialization_count
        for compiled in compiled_objects:
            _assert_scratch_free(kernel.fn.__name__, compiled)
            assert compiled.metadata.num_warps == 4
            assert compiled.metadata.shared == expected_shared
    for compiled in kernels[0].device_caches[device][0].values():
        assert "buffer_atomic_pk_add_bf16" in compiled.asm["amdgcn"]
