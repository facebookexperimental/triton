"""TLX AMD tests -- CDNA4 (gfx950)."""

import ast
import contextlib
from dataclasses import dataclass, replace
import importlib
import inspect
import itertools
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
from types import SimpleNamespace
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


def _gfx950_device_indices():
    return [
        index for index in range(torch.cuda.device_count())
        if getattr(torch.cuda.get_device_properties(index), "gcnArchName", "").startswith("gfx950")
    ]


@contextlib.contextmanager
def _activated_tlx_flash_attention_provider():
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


def _make_dense_reference_case(shape, causal, seed=0):
    """Build dense MHA forward state and FP32 reference gradients by head."""
    batch, heads, n_ctx, head_dim = shape
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(q)
    lse = torch.empty(shape[:-1], device="cuda", dtype=torch.float32)
    grads = tuple(torch.empty(shape, device="cuda", dtype=torch.float32) for _ in range(3))
    sm_scale = head_dim**-0.5
    causal_mask = torch.ones((n_ctx, n_ctx), device="cuda", dtype=torch.bool).triu(1) if causal else None

    for batch_idx in range(batch):
        for head_idx in range(heads):
            q_ref = q[batch_idx, head_idx].float().requires_grad_(True)
            k_ref = k[batch_idx, head_idx].float().requires_grad_(True)
            v_ref = v[batch_idx, head_idx].float().requires_grad_(True)
            scores = q_ref @ k_ref.mT * sm_scale
            if causal_mask is not None:
                scores = scores.masked_fill(causal_mask, float("-inf"))
            lse_ref = torch.logsumexp(scores, dim=-1)
            out_ref = torch.softmax(scores, dim=-1) @ v_ref
            reference = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), grad_out[batch_idx, head_idx].float())
            with torch.no_grad():
                out[batch_idx, head_idx].copy_(out_ref)
                lse[batch_idx, head_idx].copy_(lse_ref)
                for destination, source in zip(grads, reference, strict=True):
                    destination[batch_idx, head_idx].copy_(source)

    return ReferenceCase(q, k, v, out, grad_out, lse, sm_scale, causal, grads)


def _make_dense_gqa_reference_case(shape, *, causal=False, seed=0, sm_scale=None):
    """Build a small supported dense GQA reference case."""
    assert amd_fa_bwd._is_supported_gqa_shape(shape)
    batch, query_heads, kv_heads, n_ctx, head_dim = shape
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn((batch, query_heads, n_ctx, head_dim), generator=generator, device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, kv_heads, n_ctx, head_dim), generator=generator, device="cuda", dtype=torch.bfloat16)
    v = torch.randn((batch, kv_heads, n_ctx, head_dim), generator=generator, device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn(q.shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:-1], device="cuda", dtype=torch.float32)
    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    sm_scale = head_dim**-0.5 if sm_scale is None else sm_scale
    causal_mask = torch.ones((n_ctx, n_ctx), device="cuda", dtype=torch.bool).triu(1) if causal else None
    group_size = query_heads // kv_heads

    for batch_idx in range(batch):
        for query_head in range(query_heads):
            kv_head = query_head // group_size
            q_ref = q[batch_idx, query_head].float().requires_grad_(True)
            k_ref = k[batch_idx, kv_head].float().requires_grad_(True)
            v_ref = v[batch_idx, kv_head].float().requires_grad_(True)
            scores = q_ref @ k_ref.mT * sm_scale
            if causal_mask is not None:
                scores = scores.masked_fill(causal_mask, float("-inf"))
            lse_ref = torch.logsumexp(scores, dim=-1)
            out_ref = torch.softmax(scores, dim=-1) @ v_ref
            reference = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), grad_out[batch_idx, query_head].float())
            with torch.no_grad():
                out[batch_idx, query_head].copy_(out_ref)
                lse[batch_idx, query_head].copy_(lse_ref)
                dq[batch_idx, query_head].copy_(reference[0])
                dk[batch_idx, kv_head].add_(reference[1])
                dv[batch_idx, kv_head].add_(reference[2])

    return ReferenceCase(q, k, v, out, grad_out, lse, sm_scale, causal, (dq, dk, dv))


def _snr_db(actual, expected):
    signal = torch.linalg.vector_norm(expected.float())
    noise = torch.linalg.vector_norm(actual.float() - expected.float())
    return 20.0 * torch.log10(signal / noise).item()


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


def _assert_fp32_dq_wave8_pipeline(compiled):
    ttgir = compiled.asm["ttgir"]
    # The public route must load both full statistics planes once into shared
    # memory, then read BM16 slices instead of refetching statistics per tile.
    assert len(re.findall(r"ttg\.local_alloc[^\n]*!ttg\.memdesc<2x512xf32,", ttgir)) == 1
    cache_loads = [
        line for line in ttgir.splitlines() if "amdg.buffer_load_to_local" in line and "-> <512xf32," in line
    ]
    assert len(cache_loads) == 2
    assert all("mask =" in line and "other =" in line for line in cache_loads)
    assert len(re.findall(r"ttg\.memdesc_dynamic_subslice[^\n]*!ttg\.memdesc<16xf32,", ttgir)) >= 2

    # Follow the store's layout alias rather than depending on SSA numbering.
    # Its low three register bits are eight consecutive BF16 D elements; the
    # warp bits retain the native D32/N32 ownership of the accumulator.
    store_layouts = []
    for line in ttgir.splitlines():
        match = re.fullmatch(
            r"(#\w+) = #ttg\.linear<\{register = (\[.*\]), lane = (\[.*\]), "
            r"warp = (\[.*\]), block = \[\]\}>", line)
        if match and ast.literal_eval(match[2])[:3] == [[0, 1], [0, 2], [0, 4]]:
            if ast.literal_eval(match[4]) == [[0, 32], [32, 0]]:
                store_layouts.append(match[1])
    assert len(store_layouts) == 1
    store_type = f"tensor<64x128xbf16, {store_layouts[0]}>"
    assert sum("amdg.buffer_store" in line and store_type in line for line in ttgir.splitlines()) == 4
    assembly = compiled.asm["amdgcn"]
    assert "buffer_atomic_add_f32" in assembly
    assert "buffer_atomic_pk_add_bf16" not in assembly
    _assert_scratch_free(compiled.name, compiled)


def _assert_dk_native_column_subtiles(compiled):
    # Interleaved dK updates must own independent D64 accumulator chains.
    # A full D128 result with a selected update hides that ownership from
    # the normal scheduled-MFMA contract.
    ttir = compiled.asm["ttir"]
    dk_updates = [
        line for line in ttir.splitlines()
        if "amdg.scheduled_mfma" in line and 'accumulator "persistent" register_class "agpr"' in line
    ]
    subtiles = [line for line in dk_updates if "-> tensor<64x64xf32," in line]
    assert subtiles, "dK must carry independent native D64 accumulator chains"
    assert all("tensor<64x16xbf16," in line and "tensor<16x64xbf16," in line for line in subtiles)
    assert "output_fragment" not in ttir


def _assert_dk_column_panels_match(result, reference):
    # Check each independent column panel against the mathematical oracle;
    # a misplaced panel must not be diluted by the other panel's norm.
    for column in (0, 64):
        target = reference[..., column:column + 64].float()
        value = result[..., column:column + 64].float()
        norm = torch.linalg.vector_norm(target)
        if norm.item() == 0:
            assert torch.count_nonzero(value).item() == 0, column
        else:
            error = (torch.linalg.vector_norm(value - target) / norm).item()
            assert error < 1e-2, (column, error)


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
                mock.patch.object(torch.cuda, "device", return_value=contextlib.nullcontext()) as device_guard,
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
        device_guard.assert_called_once_with(args[1].device)
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
        with _activated_tlx_flash_attention_provider(), mock.patch.object(
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
        with _activated_tlx_flash_attention_provider(), mock.patch.object(
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
        with _activated_tlx_flash_attention_provider(), mock.patch.object(
                amd_fa_bwd,
                "fa_backward",
                wraps=amd_fa_bwd.fa_backward,
        ) as tlx_backward:
            actual = backward()
        tlx_backward.assert_not_called()
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, reference)


@unittest.skipUnless(
    flash_attention_registry_available(),
    "Need the public PyTorch registry",
)
class TestTLXFlashAttentionProviderMultiGfx950(unittest.TestCase):

    def test_direct_aten_backward_switches_to_query_device_and_restores_current_device(self):
        gfx950_devices = _gfx950_device_indices()
        if len(gfx950_devices) < 2:
            self.skipTest("Requires at least two gfx950 GPUs")

        original_device = torch.cuda.current_device()
        previous_device, query_device = gfx950_devices[:2]
        try:
            torch.cuda.set_device(previous_device)
            with torch.cuda.device(query_device):
                shape = (1, 16, 4096, 64)
                q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                k = torch.randn_like(q)
                v = torch.randn_like(q)
                grad_out = torch.randn_like(q)
                state = torch.ops.aten._scaled_dot_product_flash_attention.default(q, k, v, 0.0, False, False)
                out, lse, cum_q, cum_k, max_q, max_k, seed, offset, _ = state

            observed_devices = []

            def fake_tlx_backward(query, key, value, *_args):
                observed_devices.append(torch.cuda.current_device())
                return torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)

            self.assertEqual(torch.cuda.current_device(), previous_device)
            with _activated_tlx_flash_attention_provider(), mock.patch.object(
                    amd_fa_bwd,
                    "fa_backward",
                    side_effect=fake_tlx_backward,
            ) as tlx_backward:
                grads = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
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
                self.assertEqual(torch.cuda.current_device(), previous_device)

            tlx_backward.assert_called_once()
            self.assertEqual(observed_devices, [query_device])
            self.assertTrue(all(grad.device.index == query_device for grad in grads))
        finally:
            torch.cuda.set_device(original_device)


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
            assert not re.search(r"(?m)^[ \t]*\w*atomic\w*(?:[ \t]|$)", compiled.asm["amdgcn"])


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


def test_varlen_d128_fp32_dq_padding_falls_back_at_i32_boundary():
    batch = 1024
    q_heads = 3
    elements_per_row = q_heads * amd_fa_varlen_bwd._HEAD_DIM
    max_fp32_rows = amd_fa_varlen_bwd._I32_BUFFER_FP32_ELEMENTS // elements_per_row
    total_q = max_fp32_rows - batch * (amd_fa_varlen_bwd._BLOCK_M - 1)
    padded_bm16_elements = (total_q + batch * (amd_fa_varlen_bwd._BLOCK_M - 1)) * elements_per_row
    padded_bm32_elements = (total_q + batch * (amd_fa_varlen_bwd._WIDE_BLOCK_M - 1)) * elements_per_row

    assert padded_bm16_elements <= amd_fa_varlen_bwd._I32_BUFFER_FP32_ELEMENTS
    assert amd_fa_varlen_bwd._I32_BUFFER_FP32_ELEMENTS < padded_bm32_elements
    assert padded_bm32_elements <= amd_fa_varlen_bwd._I32_BUFFER_BF16_ELEMENTS

    fp32_pad_rows = amd_fa_varlen_bwd._select_varlen_dq_pad_rows(
        total_q=total_q,
        batch=batch,
        q_heads=q_heads,
        block_m=amd_fa_varlen_bwd._WIDE_BLOCK_M,
        block_n=amd_fa_varlen_bwd._WIDE_BLOCK_N,
        dq_atomic_fp32=True,
    )
    bf16_pad_rows = amd_fa_varlen_bwd._select_varlen_dq_pad_rows(
        total_q=total_q,
        batch=batch,
        q_heads=q_heads,
        block_m=amd_fa_varlen_bwd._WIDE_BLOCK_M,
        block_n=amd_fa_varlen_bwd._WIDE_BLOCK_N,
        dq_atomic_fp32=False,
    )

    assert fp32_pad_rows == amd_fa_varlen_bwd._BLOCK_M
    assert bf16_pad_rows == amd_fa_varlen_bwd._WIDE_BLOCK_M
    amd_fa_varlen_bwd._validate_i32_buffer_offsets(
        total_q=total_q,
        total_kv=1,
        batch=batch,
        q_heads=q_heads,
        kv_heads=1,
        dq_atomic_fp32=True,
        dq_pad_rows=fp32_pad_rows,
    )
    amd_fa_varlen_bwd._validate_i32_buffer_offsets(
        total_q=total_q,
        total_kv=1,
        batch=batch,
        q_heads=q_heads,
        kv_heads=1,
        dq_atomic_fp32=False,
        dq_pad_rows=bf16_pad_rows,
    )
    with pytest.raises(ValueError, match="padded dQ size"):
        amd_fa_varlen_bwd._validate_i32_buffer_offsets(
            total_q=total_q,
            total_kv=1,
            batch=batch,
            q_heads=q_heads,
            kv_heads=1,
            dq_atomic_fp32=True,
            dq_pad_rows=amd_fa_varlen_bwd._WIDE_BLOCK_M,
        )


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


@pytest.mark.parametrize(
    ("q_heads", "kv_heads", "kv_splits", "dq_atomic_fp32", "supply_metadata", "shifted_input"),
    (
        pytest.param(12, 4, 3, False, False, None, id="gqa3-bf16-dq"),
        pytest.param(12, 4, 3, True, False, None, id="gqa3-fp32-dq"),
        pytest.param(64, 8, 4, False, False, None, id="gqa8-bf16-dq"),
        pytest.param(64, 8, 4, True, False, None, id="gqa8-fp32-dq"),
        pytest.param(12, 4, 3, True, True, None, id="gqa3-fp32-metadata"),
        pytest.param(64, 8, 4, True, True, None, id="gqa8-fp32-metadata"),
        pytest.param(12, 4, 3, True, False, "q", id="gqa3-fp32-misaligned-q"),
        pytest.param(64, 8, 4, True, False, "q", id="gqa8-fp32-misaligned-q"),
        pytest.param(12, 4, 3, True, False, "k", id="gqa3-fp32-misaligned-k"),
        pytest.param(12, 4, 3, True, False, "v", id="gqa3-fp32-misaligned-v"),
        pytest.param(12, 4, 3, True, False, "do", id="gqa3-fp32-misaligned-do"),
        pytest.param(12, 4, 3, False, False, "q", id="gqa3-bf16-misaligned-q"),
        pytest.param(12, 4, 3, False, False, "k", id="gqa3-bf16-misaligned-k"),
        pytest.param(12, 4, 3, False, False, "v", id="gqa3-bf16-misaligned-v"),
        pytest.param(12, 4, 3, False, False, "do", id="gqa3-bf16-misaligned-do"),
        pytest.param(12, 4, 3, True, True, "q", id="gqa3-fp32-misaligned-q-metadata"),
    ),
)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_legacy_prefix_public_backward_gfx950(monkeypatch, q_heads, kv_heads, kv_splits, dq_atomic_fp32,
                                                          supply_metadata, shifted_input):
    q_lengths, kv_lengths = _make_seeded_extend_attention_lengths(batch=19, max_context=12331, seed=42)
    check_reference = dq_atomic_fp32 or shifted_input is not None
    if check_reference:
        # Genuine exact-family offsets: the first three sequences end in
        # 1/15/16 rows after two full Q512 chunks, with KV tails 1/255/full.
        # The owner must accumulate a second nonzero chunk; the fourth
        # sequence also exercises a query shorter than one Q512 chunk.
        q_lengths = [1025, 1039, 1040, 321, 5662, *([3086] * 13), 1549]
        kv_lengths = [1281, 1279, 1280, 513, 10414, *([6248] * 7), *([6247] * 6), 4711]
    total_q, total_kv = sum(q_lengths), sum(kv_lengths)
    assert len(q_lengths) == len(kv_lengths) == 19
    assert all(0 < q_len <= kv_len for q_len, kv_len in zip(q_lengths, kv_lengths, strict=True))
    assert (total_q, total_kv, max(q_lengths), max(kv_lengths)) == (50754, 100696, 5662, 10414)
    cu_q = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    cu_kv = torch.tensor([0, *torch.tensor(kv_lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    # Legacy plans have a host-known wide count. Chunking also supports the
    # device-built plan, whose valid task count is available only on device.
    metadata = (total_q, total_kv, max(q_lengths), max(kv_lengths)) if supply_metadata else ()
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv, *metadata)
    if supply_metadata:
        assert plan.wide_task_count is None
        if dq_atomic_fp32 and shifted_input is None:
            valid_count = int(plan.task_counts[amd_fa_varlen_bwd._WIDE_KV_TASK_COUNT.value].item())
            assert 0 < valid_count < plan.wide_q_len.numel() <= 512
            # Invalid capacity entries must not enter the descending-Q sort.
            plan.wide_q_len[valid_count:].fill_(plan.max_q)
    else:
        assert plan.wide_task_count is not None and plan.wide_task_count > 0

    # Inactive sequences have Q=0, V=O=1 and dO=0: uniform attention with
    # LSE=log(KV length), and all three gradients exactly zero.
    q = torch.zeros((total_q, q_heads, 128), dtype=torch.bfloat16, device="cuda")
    k = torch.ones((total_kv, kv_heads, 128), dtype=torch.bfloat16, device="cuda")
    v = torch.ones_like(k)
    out = torch.ones_like(q)
    do = torch.zeros_like(q)
    lse = torch.empty((q_heads, total_q), dtype=torch.float32, device="cuda")
    q_start = 0
    for q_length, kv_length in zip(q_lengths, kv_lengths, strict=True):
        lse[:, q_start:q_start + q_length].fill_(kv_length)
        q_start += q_length
    lse.log_()
    expected = None
    if check_reference:
        # Only four sequences need a dense independent oracle. Their inputs
        # and gradients are nonzero for every head; all other outputs are
        # checked against the analytical zero result, without dense matrices.
        checked_q_lengths, checked_kv_lengths = q_lengths[:4], kv_lengths[:4]
        checked = _make_varlen_d128_reference_case(checked_q_lengths, checked_kv_lengths, q_heads=q_heads,
                                                   kv_heads=kv_heads, seed=2473)
        checked_q, checked_kv = sum(checked_q_lengths), sum(checked_kv_lengths)
        for full, small, count in zip((q, k, v, out, do), checked[:5],
                                      (checked_q, checked_kv, checked_kv, checked_q, checked_q), strict=True):
            full[:count].copy_(small)
        lse[:, :checked_q].copy_(checked[5])
        expected = checked[-1]
        del checked
    if shifted_input is not None:
        inputs = {"q": q, "k": k, "v": v, "do": do}
        original = inputs[shifted_input]
        storage = torch.empty(original.numel() + 1, dtype=original.dtype, device=original.device)
        shifted = storage[1:].view_as(original)
        shifted.copy_(original)
        assert shifted.is_contiguous() and shifted.data_ptr() % 16 != 0
        inputs[shifted_input] = shifted
        q, k, v, do = (inputs[name] for name in ("q", "k", "v", "do"))
    else:
        assert all(tensor.data_ptr() % 16 == 0 for tensor in (q, k, v, do))
    chunked = dq_atomic_fp32 and shifted_input is None
    h12_handoff = chunked and (q_heads, kv_heads) == (12, 4)
    h64_handoff = chunked and (q_heads, kv_heads) == (64, 8) and not supply_metadata
    ds_handoff = h12_handoff or h64_handoff
    empty_like = torch.empty_like
    direct_outputs = []

    def capture_final_outputs(tensor, *args, **kwargs):
        buffer = empty_like(tensor, *args, **kwargs)
        if chunked and tensor is k:
            buffer.fill_(float("nan"))
            direct_outputs.append(buffer)
        return buffer

    monkeypatch.setattr(torch, "empty_like", capture_final_outputs)
    allocate_partials = amd_fa_varlen_bwd._allocate_varlen_dkdv_partials
    partial_workspaces = []

    def capture_partials(k, splits):
        buffers = allocate_partials(k, splits)
        partial_workspaces.append((splits, *buffers))
        return buffers

    monkeypatch.setattr(amd_fa_varlen_bwd, "_allocate_varlen_dkdv_partials", capture_partials)
    legacy_preprocess_launches = _capture_kernel_with_constexprs(monkeypatch, amd_fa_varlen_bwd,
                                                                 "_varlen_bwd_preprocess_dynamic_owner_queue")
    preprocess_launches = _capture_kernel_with_constexprs(monkeypatch, amd_fa_varlen_bwd, "_varlen_bwd_preprocess")
    rolling_launches = {
        splits:
        _capture_kernel_with_constexprs(monkeypatch, amd_fa_varlen_bwd,
                                        f"_varlen_bwd_interleaved_bm32_rolling_fp32_queue_s{splits}")
        for splits in (3, 4)
    }
    shared_launches = _capture_kernel_with_constexprs(monkeypatch, amd_fa_varlen_bwd,
                                                      "_varlen_bwd_interleaved_bm16_bn256_fp32_kernel")
    convert_launches = _capture_kernel_with_constexprs(monkeypatch, amd_fa_varlen_bwd,
                                                       "_varlen_mha_dq_convert_coalesced_fp32_swizzled_kernel")
    fallback_launches = _capture_kernel_with_constexprs(monkeypatch, amd_fa_varlen_bwd,
                                                        "_varlen_bwd_interleaved_kernel")
    generic_launches = _capture_kernel_with_constexprs(monkeypatch, amd_fa_varlen_bwd,
                                                       "_varlen_bwd_interleaved_bm32_kernel")
    reduce_launches = _capture_kernel_with_constexprs(monkeypatch, amd_fa_varlen_bwd, "_varlen_dkdv_reduce_kernel")
    producer_launches = _capture_kernel_with_constexprs(monkeypatch, amd_fa_varlen_bwd, "_prefix_h12_ds_producer")
    consumer_launches = _capture_kernel_with_constexprs(monkeypatch, amd_fa_varlen_bwd, "_prefix_h12_ds_dq_consumer")
    h64_producer_launches = _capture_kernel_with_constexprs(monkeypatch, amd_fa_varlen_bwd,
                                                            "_prefix_h64_packed_ds_producer")
    h64_consumer_launches = _capture_kernel_with_constexprs(monkeypatch, amd_fa_varlen_bwd,
                                                            "_prefix_h64_packed_ds_dq_consumer")
    helper = amd_fa_varlen_bwd._try_allocate_prefix_h12_ds
    helper_calls = []

    def observe_helper(*args, **kwargs):
        helper_calls.append(kwargs["eligible_h12"])
        return helper(*args, **kwargs)

    monkeypatch.setattr(amd_fa_varlen_bwd, "_try_allocate_prefix_h12_ds", observe_helper)
    actual = amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, 128**-0.5, dq_atomic_fp32=dq_atomic_fp32)
    torch.cuda.synchronize()

    assert not generic_launches
    assert helper_calls == ([True] if h12_handoff else [])
    if not h12_handoff:
        assert not producer_launches and not consumer_launches
    if not h64_handoff:
        assert not h64_producer_launches and not h64_consumer_launches
    if h64_handoff:
        producer_launches, consumer_launches = h64_producer_launches, h64_consumer_launches
    if shifted_input is None:
        assert not fallback_launches
    if chunked:
        assert len(preprocess_launches) == 1
        if ds_handoff:
            assert len(producer_launches) == len(consumer_launches) == 1
            assert not shared_launches and not convert_launches
        else:
            assert len(shared_launches) == len(convert_launches) == 1
        assert not partial_workspaces
        assert not reduce_launches
        assert len(direct_outputs) == 2
        assert actual[1] is direct_outputs[0]
        assert actual[2] is direct_outputs[1]
        assert not legacy_preprocess_launches
        assert all(not launches for launches in rolling_launches.values())
        preprocess_kwargs, _ = preprocess_launches[0]
        core_kwargs, compiled = (producer_launches if ds_handoff else shared_launches)[0]
        assert preprocess_kwargs["ZERO_DQ"] is (not ds_handoff)
        assert preprocess_kwargs["PACK_STATS_MHA16"] is True and preprocess_kwargs["PACK_STATS"] is False
        assert preprocess_kwargs["DQ_PAD_ROWS"] == 16
        if ds_handoff:
            consumer_kwargs, consumer = consumer_launches[0]
            assert core_kwargs["DS_CAP"] == consumer_kwargs["DS_CAP"] == 10496
            assert consumer_kwargs["TOTAL_Q_PADDED"] == 51039
            assert consumer_kwargs["enable_fp_fusion"] is False
            assert consumer_kwargs["num_warps"] == 4
            assert "buffer_atomic" not in consumer.asm["amdgcn"]
        else:
            convert_kwargs, _ = convert_launches[0]
            assert convert_kwargs["DQ_PAD_ROWS"] == 16
        assert core_kwargs["CHUNKED_Q"] is True
        assert core_kwargs["SORT_TASKS"] is True
        assert core_kwargs["Q_SPLITS"] == 1
        assert (core_kwargs["HQ"], core_kwargs["HKV"], core_kwargs["BLOCK_M"],
                core_kwargs["BLOCK_N"]) == (q_heads, kv_heads, 16, 256)
        if ds_handoff:
            assert "buffer_atomic" not in compiled.asm["amdgcn"]
        else:
            assert "buffer_atomic_add_f32" in compiled.asm["amdgcn"]
        assert "buffer_atomic_pk_add_bf16" not in compiled.asm["amdgcn"]
    elif shifted_input is not None:
        assert len(partial_workspaces) == 1
        splits, dk_partial, dv_partial = partial_workspaces[0]
        assert splits == 1 and dk_partial is None and dv_partial is None
        assert len(preprocess_launches) == 1 and len(fallback_launches) == 2
        assert not legacy_preprocess_launches and not shared_launches and not convert_launches and not reduce_launches
        assert all(not launches for launches in rolling_launches.values())
        preprocess_kwargs, _ = preprocess_launches[0]
        assert preprocess_kwargs["ZERO_DQ"] is False and preprocess_kwargs["PACK_STATS"] is False
        assert preprocess_kwargs["DQ_PAD_ROWS"] == 16
        assert {kwargs["FULL_KV_TILE"] for kwargs, _ in fallback_launches} == {False, True}
        for core_kwargs, compiled in fallback_launches:
            assert (core_kwargs["KV_SPLITS"], core_kwargs["BLOCK_M"], core_kwargs["BLOCK_N"]) == (1, 16, 128)
            assert core_kwargs["QDO_ALIGNED"] is False and core_kwargs["CACHE_MHA_STATS"] is False
            atomic = "buffer_atomic_add_f32" if dq_atomic_fp32 else "buffer_atomic_pk_add_bf16"
            assert atomic in compiled.asm["amdgcn"]
    else:
        assert len(legacy_preprocess_launches) == len(rolling_launches[kv_splits]) == 1
        assert all(not launches for splits, launches in rolling_launches.items() if splits != kv_splits)
        assert not preprocess_launches and not shared_launches and not convert_launches and not reduce_launches
        preprocess_kwargs, _ = legacy_preprocess_launches[0]
        assert preprocess_kwargs["ZERO_DQ"] is True and preprocess_kwargs["PACK_STATS"] is True
        core_kwargs, compiled = rolling_launches[kv_splits][0]
        assert core_kwargs["KV_SPLITS"] == kv_splits
        atomic = "buffer_atomic_add_f32" if dq_atomic_fp32 else "buffer_atomic_pk_add_bf16"
        assert atomic in compiled.asm["amdgcn"]
    for index, (name, result, source) in enumerate(zip(("dq", "dk", "dv"), actual, (q, k, v), strict=True)):
        assert result.shape == source.shape and result.device == source.device, name
        assert result.dtype is torch.bfloat16, name
        if check_reference:
            reference = expected[index]
            count = reference.shape[0]
            assert torch.count_nonzero(reference).item() > 0, name
            assert torch.isfinite(result[:count]).all(), name
            relative_l2 = torch.linalg.vector_norm(result[:count].float() -
                                                   reference.float()) / torch.linalg.vector_norm(reference.float())
            assert relative_l2.item() < 1e-2, (name, relative_l2.item())
            assert torch.count_nonzero(result[count:]).item() == 0, name
        else:
            assert torch.count_nonzero(result).item() == 0, name
    if check_reference:
        # Check each final chunk independently, so the two preceding Q512
        # chunks cannot conceal a missing or incorrectly rebased tail.
        q_start = 0
        for q_length in checked_q_lengths[:3]:
            tail = slice(q_start + 1024, q_start + q_length)
            reference = expected[0][tail].float()
            relative_l2 = torch.linalg.vector_norm(actual[0][tail].float() -
                                                   reference) / torch.linalg.vector_norm(reference)
            assert relative_l2.item() < 1e-2, (q_length - 1024, relative_l2.item())
            q_start += q_length
        # Check the short sequence independently; aggregate error from the
        # three longer sequences must not conceal its missing contribution.
        short_q = slice(sum(checked_q_lengths[:3]), sum(checked_q_lengths))
        short_kv = slice(sum(checked_kv_lengths[:3]), sum(checked_kv_lengths))
        for name, result, reference, rows in zip(
            ("dq", "dk", "dv"),
                actual,
                expected,
            (short_q, short_kv, short_kv),
                strict=True,
        ):
            reference = reference[rows].float()
            assert torch.count_nonzero(reference).item() > 0, name
            relative_l2 = torch.linalg.vector_norm(result[rows].float() -
                                                   reference) / torch.linalg.vector_norm(reference)
            assert relative_l2.item() < 1e-2, (name, "short-query", relative_l2.item())


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


@pytest.mark.parametrize(
    ("q_lengths", "kv_lengths", "q_heads", "kv_heads", "causal", "wide_split"),
    (
        pytest.param([17], [129], 2, 2, False, False, id="noncausal-mha"),
        pytest.param([33], [257], 3, 1, False, False, id="noncausal-gqa"),
        pytest.param([5460], [17], 3, 1, False, True, id="noncausal-gqa3-split"),
        pytest.param([17], [17], 2, 2, True, False, id="causal-mha"),
    ),
)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_fp32_dq_atomics_generic_paths_gfx950(
    q_lengths,
    kv_lengths,
    q_heads,
    kv_heads,
    causal,
    wide_split,
):
    generic_core = amd_fa_varlen_bwd._varlen_bwd_interleaved_kernel
    wide_core = amd_fa_varlen_bwd._varlen_bwd_interleaved_bm32_kernel
    exact_core = amd_fa_varlen_bwd._varlen_bwd_interleaved_bm16_bn256_fp32_kernel
    generic_core.device_caches.clear()
    wide_core.device_caches.clear()
    exact_core.device_caches.clear()

    case = _make_varlen_d128_reference_case(
        q_lengths,
        kv_lengths,
        q_heads=q_heads,
        kv_heads=kv_heads,
        seed=2399 + q_heads,
        causal=causal,
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

    actual = amd_fa_varlen_bwd.fa_varlen_backward(
        q,
        k,
        v,
        out,
        do,
        lse,
        plan,
        scale,
        causal=causal,
        dq_atomic_fp32=True,
    )
    torch.cuda.synchronize()

    for name, result, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True):
        assert result.dtype is torch.bfloat16, name
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - reference.float()) / torch.linalg.vector_norm(
            reference.float())
        assert relative_l2.item() < 1e-2, (name, relative_l2.item())

    device = torch.cuda.current_device()
    selected_core = wide_core if wide_split else generic_core
    unused_core = generic_core if wide_split else wide_core
    compiled_objects = tuple(selected_core.device_caches[device][0].values())
    assert compiled_objects
    assert not unused_core.device_caches.get(device)
    assert not exact_core.device_caches.get(device)
    atomic_assembly = "\n".join(compiled.asm["amdgcn"] for compiled in compiled_objects)
    assert "buffer_atomic_add_f32" in atomic_assembly
    assert "buffer_atomic_pk_add_bf16" not in atomic_assembly


@pytest.mark.parametrize(
    ("max_q", "q_heads", "kv_heads", "sm_scale"),
    (
        pytest.param(300, 4, 4, None, id="300"),
        pytest.param(400, 4, 4, None, id="400"),
        pytest.param(300, 12, 4, None, id="gqa3-q300"),
        pytest.param(400, 12, 4, None, id="gqa3-q400"),
        pytest.param(300, 12, 4, 0.125, id="gqa3-positive-scale"),
        pytest.param(400, 12, 4, -0.125, id="gqa3-negative-scale"),
        pytest.param(300, 12, 4, 0.0, id="gqa3-q300-zero-scale"),
        pytest.param(400, 12, 4, 0.0, id="gqa3-q400-zero-scale"),
    ),
)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_fp32_dq_atomics_gfx950(monkeypatch, max_q, q_heads, kv_heads, sm_scale):
    kernels = (
        amd_fa_varlen_bwd._varlen_bwd_preprocess,
        amd_fa_varlen_bwd._varlen_bwd_interleaved_bm16_bn256_fp32_kernel,
        amd_fa_varlen_bwd._varlen_mha_dq_convert_coalesced_fp32_swizzled_kernel,
        amd_fa_varlen_bwd._varlen_bwd_interleaved_kernel,
        amd_fa_varlen_bwd._varlen_mha_dq_convert_coalesced_kernel,
    )
    for kernel in kernels:
        kernel.device_caches.clear()

    preprocess_launches = _capture_kernel_with_constexprs(monkeypatch, amd_fa_varlen_bwd, "_varlen_bwd_preprocess")
    exact_launches = _capture_kernel_with_constexprs(monkeypatch, amd_fa_varlen_bwd,
                                                     "_varlen_bwd_interleaved_bm16_bn256_fp32_kernel")
    exact_convert_launches = _capture_kernel_with_constexprs(monkeypatch, amd_fa_varlen_bwd,
                                                             "_varlen_mha_dq_convert_coalesced_fp32_swizzled_kernel")
    generic_launches = _capture_kernel_with_constexprs(monkeypatch, amd_fa_varlen_bwd, "_varlen_bwd_interleaved_kernel")
    generic_convert_launches = _capture_kernel_with_constexprs(monkeypatch, amd_fa_varlen_bwd,
                                                               "_varlen_mha_dq_convert_coalesced_kernel")

    # Match the public route's batch/maxima while keeping the reference small.
    # Exercise short/partial BM16 tiles, both sides of Q256, and KV tails.
    # Q400 also exercises aligned runtime totals; Q300 keeps unaligned totals.
    last_q = 10 if max_q == 400 else 1
    q_lengths = [max_q, 1, 15, 16, 17, 255, 256, 257, max_q - 1, *([1] * 758), last_q]
    kv_lengths = [3200, 1, 127, 128, 129, 255, 256, 257, 257, *([1] * 759)]
    case = _make_varlen_d128_reference_case(q_lengths, kv_lengths, q_heads=q_heads, kv_heads=kv_heads, seed=2417,
                                            sm_scale=sm_scale)
    q, k, v, out, do, lse, cu_q, cu_kv, scale, expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(
        cu_q,
        cu_kv,
        q.shape[0],
        k.shape[0],
        max(q_lengths),
        max(kv_lengths),
    )
    assert plan.batch == len(q_lengths) == len(kv_lengths) == 768
    assert (plan.total_q, plan.total_kv, plan.max_q, plan.max_kv) == (sum(q_lengths), sum(kv_lengths), max_q, 3200)
    assert all(tensor.data_ptr() % 16 == 0 for tensor in (q, k, v, do))
    actual = amd_fa_varlen_bwd.fa_varlen_backward(
        q,
        k,
        v,
        out,
        do,
        lse,
        plan,
        scale,
        dq_atomic_fp32=True,
    )
    torch.cuda.synchronize()

    for name, result, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True):
        assert result.dtype is torch.bfloat16, name
        assert torch.isfinite(result).all(), name
        if sm_scale == 0.0 and name in ("dq", "dk"):
            assert torch.count_nonzero(reference).item() == 0, name
            assert torch.count_nonzero(result).item() == 0, name
            continue
        relative_l2 = torch.linalg.vector_norm(result.float() - reference.float()) / torch.linalg.vector_norm(
            reference.float())
        assert relative_l2.item() < 1e-2, (name, relative_l2.item())

    assert len(preprocess_launches) == len(exact_launches) == len(exact_convert_launches) == 1
    assert generic_launches == generic_convert_launches == []
    wide_task_count = plan.task_counts[amd_fa_varlen_bwd._WIDE_KV_TASK_COUNT.value].item()
    assert wide_task_count == 782
    wide_q_starts = plan.wide_q_start[:wide_task_count].tolist()
    assert wide_q_starts.count(0) == 13
    assert wide_q_starts.count(max_q) == 1
    preprocess_kwargs, _ = preprocess_launches[0]
    exact_kwargs, compiled = exact_launches[0]
    assert preprocess_kwargs["PACK_STATS_MHA16"] is True
    assert preprocess_kwargs["PACK_STATS"] is False
    assert exact_kwargs["HQ"] == q_heads and exact_kwargs["HKV"] == kv_heads
    assert exact_kwargs["BLOCK_M"] == 16 and exact_kwargs["BLOCK_N"] == 256
    assert exact_kwargs["reverse_local_assignment"] is True
    assert exact_kwargs["enable_sched_group_barrier_scheduler"] is False
    assert exact_kwargs["llvm_fn_attrs"] == (("amdgpu-sched-strategy", "max-ilp"), )
    assert exact_kwargs["SORT_TASKS"] is False
    assert not ({
        "sink_insts_to_avoid_spills",
        "regclass_priority_trumps_globalness",
        "disable_unclustered_high_rp_reschedule",
    } & exact_kwargs.keys())
    assembly = compiled.asm["amdgcn"]
    assert "buffer_atomic_add_f32" in assembly
    assert "buffer_atomic_pk_add_bf16" not in assembly
    _assert_dk_native_column_subtiles(compiled)
    _assert_dk_column_panels_match(actual[1], expected[1])
    if q_heads == kv_heads:
        _assert_fp32_dq_wave8_pipeline(compiled)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_fp32_dq_wave8_state_graph_replay_gfx950(monkeypatch):
    q_lengths = [400, 17, *([1] * 766)]
    kv_lengths = [3200, 257, *([1] * 766)]
    original = _make_varlen_d128_reference_case(q_lengths, kv_lengths, q_heads=4, kv_heads=4, seed=2431)
    changed = _make_varlen_d128_reference_case(q_lengths, kv_lengths, q_heads=4, kv_heads=4, seed=2437)
    q, k, v, out, do, lse, cu_q, cu_kv, scale, expected = original
    original_inputs = tuple(tensor.clone() for tensor in original[:6])
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv, q.shape[0], k.shape[0], max(q_lengths),
                                                     max(kv_lengths))
    preprocess = amd_fa_varlen_bwd._varlen_bwd_preprocess
    exact_launches = _capture_kernel_with_constexprs(monkeypatch, amd_fa_varlen_bwd,
                                                     "_varlen_bwd_interleaved_bm16_bn256_fp32_kernel")

    class PoisonedPreprocess:

        def __getitem__(self, grid):

            def launch(o, do, delta, cu_q, dq_acc, total_q_padded, task_counts, **kwargs):
                assert kwargs["ZERO_DQ"] and kwargs["PACK_STATS_MHA16"]
                assert delta.dtype is dq_acc.dtype is torch.float32
                # Graph replay reuses these allocations. Both the first call
                # and every replay must initialize statistics and dQ scratch.
                delta.fill_(float("nan"))
                dq_acc.fill_(float("nan"))
                return preprocess[grid](o, do, delta, cu_q, dq_acc, total_q_padded, task_counts, **kwargs)

            return launch

    monkeypatch.setattr(amd_fa_varlen_bwd, "_varlen_bwd_preprocess", PoisonedPreprocess())

    def backward():
        return amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale, dq_atomic_fp32=True)

    def check(actual, reference):
        for name, result, target in zip(("dq", "dk", "dv"), actual, reference, strict=True):
            assert result.dtype is torch.bfloat16, name
            assert torch.isfinite(result).all(), name
            relative_l2 = torch.linalg.vector_norm(result.float() - target.float()) / torch.linalg.vector_norm(
                target.float())
            assert relative_l2.item() < 1e-2, (name, relative_l2.item())

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        first = backward()
    torch.cuda.current_stream().wait_stream(stream)
    check(first, expected)
    assert len(exact_launches) == 1
    _assert_fp32_dq_wave8_pipeline(exact_launches[0][1])
    with torch.cuda.stream(stream):
        repeated = backward()
    torch.cuda.current_stream().wait_stream(stream)
    check(repeated, expected)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = backward()
    output_pointers = tuple(tensor.data_ptr() for tensor in actual)
    # Fresh forward output/LSE accompany the changed Q/K/V/dO. The replayed
    # launch keeps its input pointers, plan, scratch and output allocations.
    for inputs, reference in ((original_inputs, expected), (changed[:6], changed[-1]), (changed[:6], changed[-1]),
                              (original_inputs, expected)):
        for target, replacement in zip(original[:6], inputs, strict=True):
            target.copy_(replacement)
        for result in actual:
            result.fill_(float("nan"))
        graph.replay()
        check(actual, reference)
        assert tuple(tensor.data_ptr() for tensor in actual) == output_pointers


@pytest.mark.parametrize(
    ("max_q", "q_heads", "kv_heads", "dq_atomic_fp32", "shifted_input"),
    (
        pytest.param(401, 4, 4, True, None, id="q-above-cache-domain"),
        pytest.param(300, 4, 4, True, "q", id="misaligned-q"),
        pytest.param(300, 4, 4, True, "k", id="misaligned-k"),
        pytest.param(300, 4, 4, True, "v", id="misaligned-v"),
        pytest.param(300, 4, 4, True, "do", id="misaligned-do"),
        pytest.param(401, 12, 4, True, None, id="gqa3-q-above-cache-domain"),
        pytest.param(300, 12, 4, True, "q", id="gqa3-misaligned-q"),
        pytest.param(300, 12, 4, True, "k", id="gqa3-misaligned-k"),
        pytest.param(300, 12, 4, True, "v", id="gqa3-misaligned-v"),
        pytest.param(300, 12, 4, True, "do", id="gqa3-misaligned-do"),
        pytest.param(300, 12, 4, False, None, id="gqa3-bf16-dq-unchanged"),
    ),
)
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_fp32_dq_wave8_fallback_gfx950(monkeypatch, max_q, q_heads, kv_heads, dq_atomic_fp32,
                                                   shifted_input):
    q_lengths = [max_q, 17, *([1] * 766)]
    kv_lengths = [3200, 257, *([1] * 766)]
    case = list(_make_varlen_d128_reference_case(q_lengths, kv_lengths, q_heads=q_heads, kv_heads=kv_heads, seed=2441))
    if shifted_input is not None:
        index = {"q": 0, "k": 1, "v": 2, "do": 4}[shifted_input]
        tensor = case[index]
        storage = torch.empty(tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device)
        shifted = storage[1:].view(tensor.shape)
        shifted.copy_(tensor)
        assert shifted.is_contiguous() and shifted.data_ptr() % 16 != 0
        case[index] = shifted
    q, k, v, out, do, lse, cu_q, cu_kv, scale, expected = case
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv, q.shape[0], k.shape[0], max(q_lengths),
                                                     max(kv_lengths))

    class UnsupportedExactKernel:

        def __getitem__(self, grid):
            # Fail at dispatch, before an unsafe direct-to-LDS launch can run.
            pytest.fail("The FP32 BM16/BN256 core must fall back outside its input domain")

    monkeypatch.setattr(amd_fa_varlen_bwd, "_varlen_bwd_interleaved_bm16_bn256_fp32_kernel", UnsupportedExactKernel())
    generic_launches = _capture_kernel_with_constexprs(monkeypatch, amd_fa_varlen_bwd, "_varlen_bwd_interleaved_kernel")
    actual = amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, scale, dq_atomic_fp32=dq_atomic_fp32)
    torch.cuda.synchronize()

    assert generic_launches
    for name, result, reference in zip(("dq", "dk", "dv"), actual, expected, strict=True):
        assert result.dtype is torch.bfloat16, name
        assert torch.isfinite(result).all(), name
        relative_l2 = torch.linalg.vector_norm(result.float() - reference.float()) / torch.linalg.vector_norm(
            reference.float())
        assert relative_l2.item() < 1e-2, (name, relative_l2.item())


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


# Dense backward contract tests live here rather than in the packaged kernel
# module. Keep these broader than the PyTorch provider allow-list: they protect
# the retained direct API and its explicit experimental dispatches.
def test_dense_bwd_tutorial_compatibility_shim():
    from triton.language.extra.tlx.tutorials import amd_fa_bwd as tutorial_bwd
    from triton.language.extra.tlx.tutorials.amd_fa_bwd import (
        ReferenceCase as TutorialReferenceCase,
        is_hip_cdna4 as tutorial_is_hip_cdna4,
        make_reference_case as tutorial_make_reference_case,
    )

    assert tutorial_bwd.fa_backward is amd_fa_bwd.fa_backward
    assert tutorial_bwd.fa_backward_support_error is amd_fa_bwd.fa_backward_support_error
    assert tutorial_bwd.SUPPORTED_SHAPES is amd_fa_bwd.SUPPORTED_SHAPES
    assert tutorial_bwd.ReferenceCase is TutorialReferenceCase
    assert tutorial_bwd.is_hip_cdna4 is tutorial_is_hip_cdna4 is is_hip_cdna4
    assert tutorial_bwd.make_reference_case is tutorial_make_reference_case
    assert {"ReferenceCase", "is_hip_cdna4", "make_reference_case"} <= set(tutorial_bwd.__all__)

    q, k, v, out, grad_out, lse = (torch.empty(0) for _ in range(6))
    case = TutorialReferenceCase(q, k, v, out, grad_out, lse, 0.125, True, (q, k, v))
    for actual, expected in zip(case.kernel_args[:6], (q, k, v, out, grad_out, lse), strict=True):
        assert actual is expected
    assert case.kernel_args[6:] == (0.125, True)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize("causal", [False, True])
def test_dense_bwd_tutorial_make_reference_case_gfx950(causal):
    from triton.language.extra.tlx.tutorials.amd_fa_bwd import make_reference_case

    case = make_reference_case((1, 1, 8, 4), causal, seed=17)
    assert case.q.shape == (1, 1, 8, 4)
    assert case.o.dtype is torch.bfloat16
    assert case.lse.dtype is torch.float32
    assert len(case.grads) == 3
    for grad in case.grads:
        assert grad.shape == case.q.shape
        assert torch.isfinite(grad).all()
    assert case.kernel_args == (
        case.q,
        case.k,
        case.v,
        case.o,
        case.do,
        case.lse,
        case.sm_scale,
        causal,
    )


def test_dense_bwd_gqa_benchmark_shapes_match_hk_series():
    assert amd_fa_bwd.GQA_BENCHMARK_SHAPES == {
        (16, 64, 8, 1024, 128),
        (16, 64, 8, 2048, 128),
        (16, 64, 8, 4096, 128),
        (16, 64, 8, 8192, 128),
        (15, 64, 8, 16384, 128),
    }


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1, 256, 128),
        (2, 6, 3, 512, 128),
        (1, 6, 2, 768, 128),
        (1, 12, 3, 1024, 128),
        (16, 64, 8, 4096, 128),
        (1, 520, 8, 16384, 128),
    ],
)
def test_dense_bwd_gqa_supported_shape_constraint(shape):
    assert amd_fa_bwd._is_supported_gqa_shape(shape)


@pytest.mark.parametrize(
    "shape",
    [
        (0, 8, 1, 256, 128),
        (1, 0, 1, 256, 128),
        (1, 8, 0, 256, 128),
        (1, 27, 8, 256, 128),
        (1, 8, 1, 128, 128),
        (1, 8, 1, 200, 128),
        (1, 8, 1, 384, 128),
        (1, 8, 1, 256, 256),
    ],
)
def test_dense_bwd_gqa_unsupported_shape_constraint(shape):
    assert not amd_fa_bwd._is_supported_gqa_shape(shape)


def test_dense_bwd_d128_dispatches_cover_retained_topologies(monkeypatch):
    shape = (16, 27, 200, 128)
    route_envs = (
        amd_fa_bwd._D128_EXACT_ENABLE_ENV,
        amd_fa_bwd._D128_PERSISTENT_ENABLE_ENV,
        amd_fa_bwd._D128_PERSISTENT_PIPE_ENABLE_ENV,
    )
    for name in route_envs:
        monkeypatch.setenv(name, "0")

    full = amd_fa_bwd._select_d128_dispatch(shape, False)
    causal = amd_fa_bwd._select_d128_dispatch(shape, True)
    assert full.entry is causal.entry is amd_fa_bwd._attn_bwd_dkdv_d128_split_kernel
    assert (full.pipelined, full.rectangular, full.block_m, full.block_n, full.num_warps) == (True, True, 32, 64, 4)
    assert (causal.pipelined, causal.rectangular, causal.block_m, causal.block_n, causal.num_warps) == (
        False,
        False,
        32,
        32,
        2,
    )

    monkeypatch.setenv(amd_fa_bwd._D128_EXACT_ENABLE_ENV, "1")
    exact = amd_fa_bwd._select_d128_dispatch(shape, False)
    assert exact.entry is amd_fa_bwd._attn_bwd_dkdv_dq_d128_combined_kernel
    assert exact.exact and not exact.pipelined and exact.num_warps == 4

    monkeypatch.setenv(amd_fa_bwd._D128_EXACT_ENABLE_ENV, "0")
    monkeypatch.setenv(amd_fa_bwd._D128_PERSISTENT_ENABLE_ENV, "1")
    persistent = amd_fa_bwd._select_d128_dispatch(shape, False)
    assert persistent.entry is amd_fa_bwd._attn_bwd_dkdv_dq_d128_combined_kernel
    assert not persistent.exact and not persistent.pipelined and persistent.num_warps == 8

    monkeypatch.setenv(amd_fa_bwd._D128_PERSISTENT_ENABLE_ENV, "0")
    monkeypatch.setenv(amd_fa_bwd._D128_PERSISTENT_PIPE_ENABLE_ENV, "1")
    pipelined = amd_fa_bwd._select_d128_dispatch(shape, False)
    assert pipelined.entry is amd_fa_bwd._attn_bwd_dkdv_dq_d128_combined_kernel
    assert not pipelined.exact and pipelined.pipelined and pipelined.num_warps == 4

    assert amd_fa_bwd._select_d128_dispatch((1, 1, 256, 128), False).entry is \
        amd_fa_bwd._attn_bwd_dkdv_d128_split_kernel


def test_dense_bwd_d128_route_constraints_and_regalloc_options(monkeypatch):
    shape = (16, 27, 200, 128)
    assert amd_fa_bwd._d128_persistent_short_supported(shape, False)
    assert amd_fa_bwd._d128_persistent_short_supported(shape, True)
    for unsupported in (
        (1, 1, 128, 128),
        (1, 1, 256, 128),
        (1, 1, 260, 128),
        (32, 1, 2600, 256),
        (1, 1, 200, 64),
    ):
        assert not amd_fa_bwd._d128_persistent_short_supported(unsupported, False)

    assert amd_fa_bwd._select_d128_dkdv_config(shape, True) == (32, 32, 2)
    assert amd_fa_bwd._select_d128_dkdv_config(shape, False) == (32, 64, 4)
    assert amd_fa_bwd._select_d128_dkdv_config((16, 27, 260, 128), True) == (64, 64, 4)
    assert amd_fa_bwd._matrix_instr_nonkdim() == 16

    regalloc_envs = (
        amd_fa_bwd._D128_SINK_INSTS_ENV,
        amd_fa_bwd._D128_REGCLASS_PRIORITY_ENV,
        amd_fa_bwd._D128_REVERSE_LOCAL_ENV,
    )
    for name in regalloc_envs:
        monkeypatch.setenv(name, "0")
    assert not any(amd_fa_bwd._d128_regalloc_options().values())
    monkeypatch.setenv(amd_fa_bwd._D128_SINK_INSTS_ENV, "1")
    monkeypatch.setenv(amd_fa_bwd._D128_REGCLASS_PRIORITY_ENV, "1")
    assert amd_fa_bwd._d128_regalloc_options() == {
        "sink_insts_to_avoid_spills": True,
        "regclass_priority_trumps_globalness": True,
        "reverse_local_assignment": False,
    }


def test_dense_bwd_d256_dispatches_cover_retained_topologies(monkeypatch):
    monkeypatch.setenv("TLX_FA_BWD_FORCE_STAGED", "0")
    full = amd_fa_bwd._select_d256_dispatch(False)
    causal = amd_fa_bwd._select_d256_dispatch(True)
    assert full.entry is causal.entry is amd_fa_bwd._attn_bwd_dkdv_d256_producer_kernel
    assert (full.staged, full.pipelined, full.num_warps) == (False, True, 4)
    assert (causal.staged, causal.pipelined, causal.num_warps) == (False, False, 4)

    monkeypatch.setenv("TLX_FA_BWD_FORCE_STAGED", "1")
    staged = amd_fa_bwd._select_d256_dispatch(True)
    assert staged.entry is amd_fa_bwd._attn_bwd_dkdv_d256_producer_kernel
    assert (staged.staged, staged.pipelined, staged.num_warps) == (True, False, 2)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 1, 256, 128), id="group1"),
        pytest.param((1, 1, 1, 512, 128), id="group1-two-kv-tiles"),
        pytest.param((1, 2, 1, 256, 128), id="group2"),
        pytest.param((2, 3, 1, 256, 128), id="group3-batch2"),
        pytest.param((1, 4, 1, 256, 128), id="group4"),
        pytest.param((1, 8, 1, 512, 128), id="group8-two-kv-tiles"),
        pytest.param((2, 2, 2, 512, 128), id="mha-hkv2-batch2"),
        pytest.param((1, 4, 2, 256, 128), id="group2-hkv2"),
        pytest.param((2, 6, 3, 512, 128), id="group2-hkv3-batch2"),
    ],
)
def test_dense_bwd_gqa_supported_shapes_end_to_end_gfx950(shape, causal):
    case = _make_dense_gqa_reference_case(shape, causal=causal, seed=17)
    actual_grads = fa_backward(*case.kernel_args)
    for actual, expected in zip(actual_grads, case.grads, strict=True):
        assert torch.isfinite(actual).all()
        assert _snr_db(actual, expected) >= 40.0


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize("sm_scale", [0.0, -0.125], ids=["zero-scale", "negative-scale"])
def test_dense_bwd_gqa_causal_scale_edges_gfx950(sm_scale):
    case = _make_dense_gqa_reference_case((1, 2, 1, 256, 128), causal=True, seed=17, sm_scale=sm_scale)
    actual_grads = fa_backward(*case.kernel_args)
    for actual, expected in zip(actual_grads, case.grads, strict=True):
        assert torch.isfinite(actual).all()
        if torch.count_nonzero(expected).item() == 0:
            assert torch.count_nonzero(actual).item() == 0
        else:
            assert _snr_db(actual, expected) >= 40.0


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("route", ["persistent", "pipelined"], ids=["persistent", "persistent-pipeline"])
def test_dense_bwd_d128_experimental_routes_gfx950(causal, route, monkeypatch):
    options = {
        amd_fa_bwd._D128_EXACT_ENABLE_ENV: "0",
        amd_fa_bwd._D128_PERSISTENT_ENABLE_ENV: str(int(route == "persistent")),
        amd_fa_bwd._D128_PERSISTENT_PIPE_ENABLE_ENV: str(int(route == "pipelined")),
        amd_fa_bwd._D128_SINK_INSTS_ENV: "0",
        amd_fa_bwd._D128_REGCLASS_PRIORITY_ENV: "0",
        amd_fa_bwd._D128_REVERSE_LOCAL_ENV: "0",
    }
    with mock.patch.dict(os.environ, options):
        case = _make_dense_reference_case((16, 27, 200, 128), causal, seed=19)
        actual_grads = fa_backward(*case.kernel_args)
    for actual, expected in zip(actual_grads, case.grads, strict=True):
        assert torch.isfinite(actual).all()
        assert _snr_db(actual, expected) >= 40.0


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_dense_bwd_d128_causal_tail_is_repeatable_gfx950(monkeypatch):
    options = {
        amd_fa_bwd._D128_EXACT_ENABLE_ENV: "0",
        amd_fa_bwd._D128_PERSISTENT_ENABLE_ENV: "0",
        amd_fa_bwd._D128_PERSISTENT_PIPE_ENABLE_ENV: "0",
        amd_fa_bwd._D128_SINK_INSTS_ENV: "0",
        amd_fa_bwd._D128_REGCLASS_PRIORITY_ENV: "0",
        amd_fa_bwd._D128_REVERSE_LOCAL_ENV: "0",
    }
    case = _make_dense_reference_case((16, 27, 200, 128), True, seed=21)
    with mock.patch.dict(os.environ, options):
        for _ in range(5):
            actual_grads = fa_backward(*case.kernel_args)
            for actual, expected in zip(actual_grads, case.grads, strict=True):
                assert torch.isfinite(actual).all()
                assert _snr_db(actual, expected) >= 40.0


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_dense_bwd_d256_end_to_end_gfx950(causal):
    case = _make_dense_reference_case((32, 1, 2600, 256), causal, seed=23)
    actual_grads = fa_backward(*case.kernel_args)
    for actual, expected in zip(actual_grads, case.grads, strict=True):
        assert torch.isfinite(actual).all()
        assert _snr_db(actual, expected) >= 40.0


# Exact-total fixtures keep the production prefix predicate active while limiting
# dense independent references to a short prefix of nonzero sequences.
_H12_DS_Q_LENGTHS = [1025, 1039, 1040, 321, 5662, *([3086] * 13), 1549]
_H12_DS_KV_LENGTHS = [1281, 1279, 1280, 513, 10414, *([6248] * 7), *([6247] * 6), 4711]
_H12_DS_SCALES = pytest.mark.parametrize("sm_scale", (128**-0.5, 0.0, -128**-0.5), ids=("default", "zero", "negative"))
_H12_DS_METADATA = pytest.mark.parametrize("supply_metadata", (False, True), ids=("legacy", "device"))
_H12_DS_ROLES = (
    "_varlen_bwd_preprocess",
    "_prefix_h12_ds_producer",
    "_varlen_dkdv_reduce_kernel",
    "_prefix_h12_ds_dq_consumer",
    "_varlen_bwd_interleaved_bm16_bn256_fp32_kernel",
    "_varlen_mha_dq_convert_coalesced_fp32_swizzled_kernel",
)


def _make_prefix_ds_case(supply_metadata, sm_scale=128**-0.5, *, seed=2473, q_lengths=None, kv_lengths=None, active=4,
                         q_heads=12, kv_heads=4):
    q_lengths = list(_H12_DS_Q_LENGTHS if q_lengths is None else q_lengths)
    kv_lengths = list(_H12_DS_KV_LENGTHS if kv_lengths is None else kv_lengths)
    assert len(q_lengths) == len(kv_lengths) == 19
    assert (sum(q_lengths), sum(kv_lengths), max(q_lengths), max(kv_lengths)) == (50754, 100696, 5662, 10414)
    q = torch.zeros((50754, q_heads, 128), dtype=torch.bfloat16, device="cuda")
    k = torch.ones((100696, kv_heads, 128), dtype=torch.bfloat16, device="cuda")
    v, out, do = torch.ones_like(k), torch.ones_like(q), torch.zeros_like(q)
    lse = torch.empty((q_heads, 50754), dtype=torch.float32, device="cuda")
    begin = 0
    for q_len, kv_len in zip(q_lengths, kv_lengths, strict=True):
        lse[:, begin:begin + q_len].fill_(kv_len)
        begin += q_len
    lse.log_()
    reference = _make_varlen_d128_reference_case(q_lengths[:active], kv_lengths[:active], q_heads=q_heads,
                                                 kv_heads=kv_heads, seed=seed, sm_scale=sm_scale)
    nq, nk = sum(q_lengths[:active]), sum(kv_lengths[:active])
    for full, small, count in zip((q, k, v, out, do), reference[:5], (nq, nk, nk, nq, nq), strict=True):
        full[:count].copy_(small)
    lse[:, :nq].copy_(reference[5])
    cu_q = torch.tensor([0, *itertools.accumulate(q_lengths)], dtype=torch.int32, device="cuda")
    cu_kv = torch.tensor([0, *itertools.accumulate(kv_lengths)], dtype=torch.int32, device="cuda")
    metadata = (50754, 100696, 5662, 10414) if supply_metadata else ()
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv, *metadata)
    if supply_metadata:
        assert plan.wide_task_count is None
        valid = int(plan.task_counts[amd_fa_varlen_bwd._WIDE_KV_TASK_COUNT.value].item())
        assert 0 < valid < plan.wide_q_len.numel() <= 512
        plan.wide_q_len[valid:].fill_(plan.max_q)
    else:
        assert plan.wide_task_count is not None
    return (q, k, v, out, do, lse, plan), reference[-1], q_lengths[:active], kv_lengths[:active]


def _assert_prefix_ds_result(actual, inputs, expected, q_lengths, kv_lengths):
    for name, result, source, reference in zip(("dq", "dk", "dv"), actual, inputs[:3], expected, strict=True):
        assert result.shape == source.shape and result.device == source.device, name
        assert result.dtype is torch.bfloat16 and torch.isfinite(result).all(), name
        count = reference.shape[0]
        assert torch.count_nonzero(result[count:]).item() == 0, name
        lengths = q_lengths if name == "dq" else kv_lengths
        start = 0
        for length in lengths:
            target = reference[start:start + length].float()
            value = result[start:start + length].float()
            norm = torch.linalg.vector_norm(target)
            if norm.item() == 0:
                assert torch.count_nonzero(value).item() == 0, (name, start)
            else:
                assert (torch.linalg.vector_norm(value - target) / norm).item() < 1e-2, (name, start)
            if name == "dk":
                _assert_dk_column_panels_match(value, target)
            # Each producer Q512 chunk and its final tail is checked separately.
            if name == "dq":
                for offset in range(0, length, 512):
                    tail = target[offset:offset + 512]
                    tail_norm = torch.linalg.vector_norm(tail)
                    if tail_norm.item():
                        error = torch.linalg.vector_norm(value[offset:offset + 512] - tail) / tail_norm
                        assert error.item() < 1e-2, (start, offset, error.item())
            start += length
        assert start == count


def _observe_prefix_ds_kernels(monkeypatch, before=None, *, roles=_H12_DS_ROLES):
    records = []

    class ObservedKernel:

        def __init__(self, name, kernel):
            self.name, self.kernel = name, kernel

        def __getitem__(self, grid):

            def launch(*args, **kwargs):
                if before is not None:
                    before(self.name, args, kwargs)
                compiled = self.kernel[grid](*args, **kwargs)
                records.append((self.name, kwargs, compiled))
                return compiled

            return launch

    for name in roles:
        monkeypatch.setattr(amd_fa_varlen_bwd, name, ObservedKernel(name, getattr(amd_fa_varlen_bwd, name)))
    return records


def _assert_h12_ds_route(records, handoff):
    expected = [_H12_DS_ROLES[0], _H12_DS_ROLES[1], _H12_DS_ROLES[3]
                ] if handoff else [_H12_DS_ROLES[0], _H12_DS_ROLES[4], _H12_DS_ROLES[2], _H12_DS_ROLES[5]]
    assert [name for name, _, _ in records] == expected
    preprocess = records[0][1]
    assert preprocess["ZERO_DQ"] is (not handoff)
    assert preprocess["PACK_STATS_MHA16"] is True and preprocess["PACK_STATS"] is False
    assert records[1][1]["Q_SPLITS"] == (1 if handoff else 2)
    _assert_dk_native_column_subtiles(records[1][2])
    if not handoff:
        assert records[2][1]["KV_SPLITS"] == 2
    if handoff:
        assert records[-1][1]["enable_fp_fusion"] is False
        for name, _, compiled in (records[1], records[-1]):
            assert "buffer_atomic" not in compiled.asm["amdgcn"], name
    else:
        assert "buffer_atomic_add_f32" in records[1][2].asm["amdgcn"]


def _poison_h12_ds_partials(monkeypatch):
    allocate = amd_fa_varlen_bwd._allocate_varlen_dkdv_partials
    recorded = []

    def poisoned(k, splits):
        buffers = allocate(k, splits)
        assert splits == 2
        for buffer in buffers:
            assert buffer is not None and buffer.dtype is torch.float32
            buffer.fill_(float("nan"))
        recorded.append(buffers)
        return buffers

    monkeypatch.setattr(amd_fa_varlen_bwd, "_allocate_varlen_dkdv_partials", poisoned)
    return recorded


def _assert_h12_ds_empty_owners(buffers, q_lengths, kv_lengths):
    start = 0
    for q_len, kv_len in zip(q_lengths, kv_lengths, strict=True):
        chunks = (q_len + 511) // 512
        if chunks < 2:
            for buffer in buffers:
                assert torch.count_nonzero(buffer[start:start + kv_len, :, chunks:]).item() == 0
        start += kv_len


@_H12_DS_METADATA
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_h12_ds_allocation_fallback_gfx950(monkeypatch, supply_metadata):
    inputs, expected, q_lengths, kv_lengths = _make_prefix_ds_case(supply_metadata)
    empty = torch.empty
    attempts, scratch = [], []

    def injected_empty(*args, **kwargs):
        shape = tuple(args[0]) if args and isinstance(args[0], (tuple, list)) else None
        if shape == (12, 51039, 10496) and kwargs.get("dtype") is torch.bfloat16:
            attempts.append(shape)
            raise torch.cuda.OutOfMemoryError("injected optional dS allocation failure")
        value = empty(*args, **kwargs)
        if shape == (12, 51039, 128) and kwargs.get("dtype") is torch.float32:
            value.fill_(float("nan"))
            scratch.append(value)
        return value

    monkeypatch.setattr(torch, "empty", injected_empty)
    partials = _poison_h12_ds_partials(monkeypatch)
    records = _observe_prefix_ds_kernels(monkeypatch)
    actual = amd_fa_varlen_bwd.fa_varlen_backward(*inputs, 128**-0.5, dq_atomic_fp32=True)
    _assert_h12_ds_route(records, False)
    assert attempts == [(12, 51039, 10496)] and len(scratch) == len(partials) == 1
    _assert_prefix_ds_result(actual, inputs, expected, q_lengths, kv_lengths)
    _assert_h12_ds_empty_owners(partials[0], q_lengths, kv_lengths)


@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_h12_ds_kernel_error_propagates_gfx950(monkeypatch):
    inputs, _, _, _ = _make_prefix_ds_case(False)
    failure = RuntimeError("injected producer failure")
    observed = []

    def fail_producer(name, args, kwargs):
        observed.append(name)
        if name == "_prefix_h12_ds_producer":
            raise failure

    _observe_prefix_ds_kernels(monkeypatch, fail_producer)
    with pytest.raises(RuntimeError) as caught:
        amd_fa_varlen_bwd.fa_varlen_backward(*inputs, 128**-0.5, dq_atomic_fp32=True)
    assert caught.value is failure
    assert observed == ["_varlen_bwd_preprocess", "_prefix_h12_ds_producer"]


@_H12_DS_METADATA
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_h12_ds_padding_gfx950(monkeypatch, supply_metadata):
    inputs, expected, q_lengths, kv_lengths = _make_prefix_ds_case(supply_metadata)
    empty = torch.empty
    poisoned = []

    def poison_ds(*args, **kwargs):
        value = empty(*args, **kwargs)
        if value.shape == (12, 51039, 10496) and value.dtype is torch.bfloat16:
            value.fill_(float("nan"))
            poisoned.append(True)
        return value

    checked = []

    def check_padding(name, args, kwargs):
        if name != "_prefix_h12_ds_dq_consumer":
            return
        ds = args[0]
        q_start = 0
        zero_count = gap_count = guard_count = kv_padding_count = 0
        for batch, (q_len, kv_len) in enumerate(zip(_H12_DS_Q_LENGTHS, _H12_DS_KV_LENGTHS, strict=True)):
            scratch = q_start + batch * 15
            q_allocated, kv_allocated = (q_len + 15) // 16 * 16, (kv_len + 255) // 256 * 256
            block = ds[:, scratch + q_allocated - 16:scratch + q_allocated, :].reshape(12, 10496, 16)
            first_padding = q_len % 16 or 16
            q_padding = block[:, :kv_allocated, first_padding:]
            assert torch.count_nonzero(q_padding).item() == 0
            kv_padding = block[:, kv_len:kv_allocated, :first_padding]
            assert torch.count_nonzero(kv_padding).item() == 0
            gap = ds[:, scratch + q_allocated:q_start + q_len + 15 * (batch + 1), :]
            guard = block[:, kv_allocated:, :]
            assert torch.isnan(gap).all() and torch.isnan(guard).all()
            zero_count += q_padding.numel()
            gap_count += gap.numel()
            guard_count += guard.numel()
            kv_padding_count += kv_padding.numel()
            q_start += q_len
        assert min(zero_count, gap_count, guard_count, kv_padding_count) > 0
        checked.append(True)

    monkeypatch.setattr(torch, "empty", poison_ds)
    records = _observe_prefix_ds_kernels(monkeypatch, check_padding)
    actual = amd_fa_varlen_bwd.fa_varlen_backward(*inputs, 128**-0.5, dq_atomic_fp32=True)
    assert poisoned == checked == [True]
    _assert_h12_ds_route(records, True)
    _assert_prefix_ds_result(actual, inputs, expected, q_lengths, kv_lengths)


@_H12_DS_SCALES
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_h12_ds_consumer_boundaries_gfx950(sm_scale):
    q_lengths = (127, 128, 129, 511, 512, 513, 1, 17, 33)
    kv_lengths = (1, 63, 64, 65, 255, 256, 257, 513, 0)
    q_offsets, kv_offsets = [0, *itertools.accumulate(q_lengths)], [0, *itertools.accumulate(kv_lengths)]
    padded = q_offsets[-1] + 15 * len(q_lengths) + 16
    ds = torch.full((12, padded, 10496), float("nan"), dtype=torch.bfloat16, device="cuda")
    k = torch.full((max(1, kv_offsets[-1]) + 64, 4, 128), float("nan"), dtype=torch.bfloat16, device="cuda")
    dq = torch.full((q_offsets[-1] + 128, 12, 128), float("nan"), dtype=torch.bfloat16, device="cuda")
    expected = torch.zeros_like(dq[:q_offsets[-1]])
    # Each sequence activates a different head, including all three siblings
    # of one KV head. Unselected heads must remain exactly zero.
    for batch, (q_len, kv_len) in enumerate(zip(q_lengths, kv_lengths, strict=True)):
        allocated = (q_len + 15) // 16 * 16
        kv_allocated = (kv_len + 255) // 256 * 256
        scratch = q_offsets[batch] + batch * 15
        blocks = ds[:, scratch:scratch + allocated].reshape(12, allocated // 16, 10496, 16)
        blocks[:, :, :kv_allocated].zero_()
        active_head = (0, 1, 2, 3, 5, 8, 11, 7, 10)[batch]
        # Assign through physical Q16 blocks to retain the producer's ABI.
        for block in range(allocated // 16):
            valid = min(16, q_len - block * 16)
            blocks[active_head, block, :kv_len, :valid].fill_(0.25)
        if kv_len:
            for head in range(4):
                k[kv_offsets[batch]:kv_offsets[batch + 1], head].fill_(head + 1)
        total = torch.zeros((), dtype=torch.float32, device="cuda")
        for start in range(0, kv_len, 256):
            partial = torch.tensor(
                min(256, kv_len - start) * 0.25 * (active_head // 3 + 1), dtype=torch.float32, device="cuda")
            total = total + partial * sm_scale
        expected[q_offsets[batch]:q_offsets[batch + 1], active_head].fill_(total.to(torch.bfloat16))
    cu_q = torch.tensor(q_offsets, dtype=torch.int32, device="cuda")
    cu_kv = torch.tensor(kv_offsets, dtype=torch.int32, device="cuda")
    counts = torch.zeros(6, dtype=torch.int32, device="cuda")
    amd_fa_varlen_bwd._prefix_h12_ds_dq_consumer[((max(q_lengths) + 127) // 128, 12,
                                                  len(q_lengths))](ds, k, cu_q, cu_kv, counts, dq,
                                                                   TOTAL_Q_PADDED=padded, SM_SCALE=sm_scale,
                                                                   DS_CAP=10496, PLAN_ERROR_INDEX=4, num_warps=4,
                                                                   matrix_instr_nonkdim=16, enable_fp_fusion=False)
    assert torch.equal(dq[:q_offsets[-1]], expected)
    assert torch.isnan(dq[q_offsets[-1]:]).all() and torch.isnan(k[kv_offsets[-1]:]).all()
    for batch, (q_len, kv_len) in enumerate(zip(q_lengths, kv_lengths, strict=True)):
        allocated = (q_len + 15) // 16 * 16
        kv_allocated = (kv_len + 255) // 256 * 256
        scratch = q_offsets[batch] + batch * 15
        blocks = ds[:, scratch:scratch + allocated].reshape(12, allocated // 16, 10496, 16)
        assert torch.isnan(blocks[:, :, kv_allocated:]).all()
        gap_end = q_offsets[batch + 1] + 15 * (batch + 1)
        assert torch.isnan(ds[:, scratch + allocated:gap_end]).all()
    assert torch.isnan(ds[:, q_offsets[-1] + 15 * len(q_lengths):]).all()


@pytest.mark.parametrize("fixture", ("wrap", "q32_q64", "long"))
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_h12_ds_producer_boundaries_gfx950(monkeypatch, fixture):
    if fixture == "wrap":
        q_lengths = [1, 15, 16, 17, 31, 32, 127, 128, 129, 1023, 5662, 5223, 5479, 5479, 5479, 5479, 5478, 5478, 5478]
        kv_lengths = [
            10241, 767, 768, 769, 767, 768, 769, 767, 768, 1281, 10414, 5526, 5525, 10261, 10261, 10261, 10261, 10261,
            10261
        ]
    elif fixture == "q32_q64":
        q_lengths = [1, 15, 16, 17, 33, 32, 63, 64, 65, 1023, 5662, 5223, 5479, 5479, 5479, 5479, 5542, 5542, 5540]
        kv_lengths = [
            10241, 767, 768, 769, 767, 768, 769, 767, 768, 1281, 10414, 5526, 5525, 10261, 10261, 10261, 10261, 10261,
            10261
        ]
    else:
        q_lengths = [129, 15, 16, 17, 33, 32, 63, 64, 65, 1023, 5662, 5223, 5479, 5479, 5479, 5479, 5542, 5542, 5412]
        kv_lengths = [
            10414, 767, 768, 769, 767, 768, 769, 767, 768, 1281, 10414, 5398, 5480, 10261, 10261, 10261, 10261, 10261,
            10261
        ]
    inputs, expected, active_q, active_kv = _make_prefix_ds_case(fixture != "wrap", q_lengths=q_lengths,
                                                                 kv_lengths=kv_lengths, active=10)
    partials = _poison_h12_ds_partials(monkeypatch)
    records = _observe_prefix_ds_kernels(monkeypatch)
    actual = amd_fa_varlen_bwd.fa_varlen_backward(*inputs, 128**-0.5, dq_atomic_fp32=True)
    _assert_h12_ds_route(records, True)
    _assert_prefix_ds_result(actual, inputs, expected, active_q, active_kv)
    assert partials == []


@_H12_DS_METADATA
@_H12_DS_SCALES
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_h12_ds_scaled_partial_overflow_gfx950(monkeypatch, supply_metadata, sm_scale):
    q_lengths = [1025, 1039, 256, 321, 5662, 3870, *([3086] * 12), 1549]
    kv_lengths = [2049, 1279, 512, 513, 10414, *([6248] * 7), *([6247] * 6), 4711]
    assert (len(q_lengths), len(kv_lengths), sum(q_lengths), sum(kv_lengths)) == (19, 19, 50754, 100696)
    q = torch.zeros((50754, 12, 128), dtype=torch.bfloat16, device="cuda")
    k = torch.zeros((100696, 4, 128), dtype=torch.bfloat16, device="cuda")
    v, out, do = torch.zeros_like(k), torch.zeros_like(q), torch.zeros_like(q)
    lse = torch.empty((12, 50754), dtype=torch.float32, device="cuda")
    q_offsets, kv_offsets = [0, *itertools.accumulate(q_lengths)], [0, *itertools.accumulate(kv_lengths)]
    for begin, end, kv_len in zip(q_offsets[:-1], q_offsets[1:], kv_lengths, strict=True):
        lse[:, begin:end].fill_(math.log(kv_len))
    q0, q1, k0, k1 = q_offsets[2], q_offsets[3], kv_offsets[2], kv_offsets[3]
    magnitude = float(9 * 2**118)
    do[q0:q1].fill_(1)
    k[k0:k0 + 256].fill_(magnitude)
    k[k0 + 256:k1].fill_(-magnitude)
    v[k0:k0 + 256].fill_(1)
    v[k0 + 256:k1].fill_(-1)
    cu_q = torch.tensor(q_offsets, dtype=torch.int32, device="cuda")
    cu_kv = torch.tensor(kv_offsets, dtype=torch.int32, device="cuda")
    metadata = (50754, 100696, 5662, 10414) if supply_metadata else ()
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv, *metadata)
    records = _observe_prefix_ds_kernels(monkeypatch)
    actual = amd_fa_varlen_bwd.fa_varlen_backward(q, k, v, out, do, lse, plan, sm_scale, dq_atomic_fp32=True)
    _assert_h12_ds_route(records, True)
    f32 = lambda value: struct.unpack("<f", struct.pack("<f", value))[0]
    partial = f32(256 * 0.25 * magnitude)
    total = 0.0
    for _ in range(2):
        total = f32(total + f32(partial * f32(sm_scale)))
    if sm_scale > 0:
        assert total == 3.3836619134423587e37
    for name, value, source, begin, end, expected in zip(
        ("dq", "dk", "dv"),
            actual,
        (q, k, v),
        (q0, k0, k0),
        (q1, k1, k1),
        (total, 0.0, 1.5),
            strict=True,
    ):
        assert value.shape == source.shape and value.device == source.device and value.dtype is torch.bfloat16
        assert torch.isfinite(value).all(), name
        assert torch.equal(value[begin:end], torch.full_like(value[begin:end], expected)), name
        assert torch.count_nonzero(value[:begin]).item() == torch.count_nonzero(value[end:]).item() == 0, name


@pytest.mark.parametrize(("q_heads", "kv_heads", "supply_metadata"), ((12, 4, False), (12, 4, True), (64, 8, False)))
@_H12_DS_SCALES
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_prefix_ds_cold_capture_graph_replay_gfx950(monkeypatch, q_heads, kv_heads, supply_metadata,
                                                                sm_scale, tmp_path, request):
    # One fresh process and cache per metadata/scale case. Running this test
    # after other tests cannot make its fallback route accidentally warm.
    child_key = request.node.name
    if os.environ.get("TLX_PREFIX_DS_COLD_CAPTURE_CHILD") != child_key:
        cache = tmp_path / "triton-cache"
        assert not cache.exists()
        env = os.environ.copy()
        env.update(TRITON_CACHE_DIR=str(cache), TLX_PREFIX_DS_COLD_CAPTURE_CHILD=child_key)
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-s", "--tb=short",
             str(Path(__file__).resolve()) + "::" + child_key],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=1200,
        )
        assert completed.returncode == 0, completed.stdout
        return

    cache = Path(os.environ["TRITON_CACHE_DIR"])
    h12 = q_heads == 12
    roles = _H12_DS_ROLES if h12 else _H64_DS_ROLES
    assert_route = _assert_h12_ds_route if h12 else _assert_h64_ds_route
    old_names = (_H12_DS_ROLES[2], *_H12_DS_ROLES[4:]) if h12 else _H64_DS_ROLES[3:5]
    originals = {name: getattr(amd_fa_varlen_bwd, name) for name in old_names}

    def assert_cold():
        for name, function in originals.items():
            assert not any(value[0] for value in function.device_caches.values()), name
            assert not list(cache.rglob(name + ".*")), name

    assert_cold()
    inputs, expected, q_lengths, kv_lengths = _make_prefix_ds_case(supply_metadata, sm_scale, q_heads=q_heads,
                                                                   kv_heads=kv_heads)
    changed = _make_varlen_d128_reference_case(q_lengths, kv_lengths, q_heads=q_heads, kv_heads=kv_heads, seed=2479,
                                               sm_scale=sm_scale)
    q, k, v, out, do, lse, plan = inputs
    originals_active = tuple(
        tensor[:count].clone()
        for tensor, count in zip((q, k, v, out, do), (sum(q_lengths), sum(kv_lengths), sum(kv_lengths), sum(q_lengths),
                                                      sum(q_lengths)), strict=True))
    original_lse = lse[:, :sum(q_lengths)].clone()
    optional_shape = (12, 51039, 10496) if h12 else (64, plan.packed_ds_head_elements)
    phase = ["warmup"]
    empty = torch.empty
    allocations, scratch = [], []

    def observe_empty(*args, **kwargs):
        shape = tuple(args[0]) if args and isinstance(args[0], (tuple, list)) else None
        if shape == optional_shape and kwargs.get("dtype") is torch.bfloat16:
            allocations.append(phase[0])
        return empty(*args, **kwargs)

    def before(name, args, kwargs):
        if phase[0] == "capture" and name == "_varlen_bwd_preprocess":
            assert kwargs["ZERO_DQ"] is True
            delta, dq_scratch = args[2], args[4]
            assert dq_scratch.shape == (q_heads, 51039, 128) and dq_scratch.dtype is torch.float32
            # These fills are graph nodes, so poison precedes preprocessing
            # on every replay rather than only during Python capture.
            delta.fill_(float("nan"))
            dq_scratch.fill_(float("nan"))
            scratch.append((delta, dq_scratch))

    monkeypatch.setattr(torch, "empty", observe_empty)
    partials = _poison_h12_ds_partials(monkeypatch) if h12 else []
    records = _observe_prefix_ds_kernels(monkeypatch, before, roles=roles)

    def backward():
        return amd_fa_varlen_bwd.fa_varlen_backward(*inputs, sm_scale, dq_atomic_fp32=True)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        warm = backward()
    torch.cuda.current_stream().wait_stream(stream)
    assert_route(records, True)
    _assert_prefix_ds_result(warm, inputs, expected, q_lengths, kv_lengths)
    assert partials == []
    assert allocations == ["warmup"]
    del warm
    assert_cold()
    records.clear()
    phase[0] = "capture"
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = backward()
    assert_route(records, False)
    assert allocations == ["warmup"] and len(scratch) == 1 and len(partials) == (1 if h12 else 0)
    pointers = [value.data_ptr() for value in actual]
    phase[0] = "replay"
    for use_changed in (True, False, True):
        source = changed[:5] if use_changed else originals_active
        for full, small in zip((q, k, v, out, do), source, strict=True):
            full[:small.shape[0]].copy_(small)
        lse[:, :sum(q_lengths)].copy_(changed[5] if use_changed else original_lse)
        for value in actual:
            value.fill_(float("nan"))
        graph.replay()
        _assert_prefix_ds_result(actual, inputs, changed[-1] if use_changed else expected, q_lengths, kv_lengths)
        if h12:
            _assert_h12_ds_empty_owners(partials[0], q_lengths, kv_lengths)
        assert [value.data_ptr() for value in actual] == pointers


class _H12DSHostTensor:

    def __init__(self, shape, dtype=torch.bfloat16, device="cuda:3", ptr=16):
        self.shape, self.dtype, self.device, self.ptr = tuple(shape), dtype, device, ptr

    def data_ptr(self):
        return self.ptr

    def numel(self):
        result = 1
        for size in self.shape:
            result *= size
        return result

    def stride(self, axis):
        result = 1
        for size in self.shape[axis + 1:]:
            result *= size
        return result


class _H12DSHostCuda:
    OutOfMemoryError = torch.cuda.OutOfMemoryError

    def __init__(self):
        self.current_device = "cuda:0"
        self.streams = {"cuda:0": "caller_stream", "cuda:3": "input_stream"}
        self.capturing = {"cuda:0": False, "cuda:3": False}
        self.events = []
        self.errors = {}

    def device(self, target):
        cuda = self

        class Guard:

            def __enter__(self):
                self.previous = cuda.current_device
                cuda.events.append(("enter", target))
                if "enter" in cuda.errors:
                    raise cuda.errors["enter"]
                cuda.current_device = target

            def __exit__(self, *exc):
                cuda.current_device = self.previous
                cuda.events.append(("restore", self.previous))
                if "exit" in cuda.errors:
                    raise cuda.errors["exit"]
                return False

        return Guard()

    def is_current_stream_capturing(self):
        self.events.append(("query", self.current_device, self.streams[self.current_device]))
        if "query" in self.errors:
            raise self.errors["query"]
        return self.capturing[self.current_device]


class _H12DSHostTorch:
    bfloat16 = torch.bfloat16
    float32 = torch.float32
    int32 = torch.int32

    def __init__(self):
        self.cuda = _H12DSHostCuda()
        self.allocations = []
        self.errors = {}

    def _allocate(self, kind, shape, dtype, device):
        tensor = _H12DSHostTensor(shape, dtype, device)
        self.allocations.append((kind, tensor, self.cuda.current_device))
        occurrence = sum(allocated.shape == tensor.shape for _, allocated, _ in self.allocations)
        error = self.errors.get((tensor.shape, occurrence))
        if error is not None:
            raise error
        return tensor

    def empty(self, shape, *, dtype, device):
        return self._allocate("empty", shape, dtype, device)

    def zeros(self, shape, *, dtype, device):
        return self._allocate("zeros", shape, dtype, device)

    def empty_like(self, tensor):
        return self._allocate("empty_like", tensor.shape, tensor.dtype, tensor.device)


def _h12_ds_host_fixture(monkeypatch, *, case="h12", task_capacity=400):
    """Run the real host dispatcher with metadata tensors and recording launches."""
    api = _H12DSHostTorch()
    heads, kv_heads = (64, 8) if case == "h64" else (12, 4)
    if case == "causal":
        kv_heads = heads
    total_q = 50755 if case == "total_q" else 50754
    total_kv = 100697 if case == "total_kv" else 100696
    q, k, v, o, do = [
        _H12DSHostTensor(shape, ptr=18 if case == "shifted_" + name else 16) for name, shape in (
            ("q", (total_q, heads, 128)),
            ("k", (total_kv, kv_heads, 128)),
            ("v", (total_kv, kv_heads, 128)),
            ("o", (total_q, heads, 128)),
            ("do", (total_q, heads, 128)),
        )
    ]
    plan = SimpleNamespace(
        batch=20 if case == "batch" else 19,
        max_q=5663 if case == "max_q" else 5662,
        max_kv=10415 if case == "max_kv" else 10414,
        wide_task_count=None,
        task_counts=object(),
        cu_seqlens_q=object(),
        cu_seqlens_k=object(),
        dq_full_kv_sequence=None,
        dq_full_kv_start=None,
    )
    for name in (
            "wide_kv_start",
            "wide_q_start",
            "wide_dq_start",
            "wide_q_len",
            "wide_kv_valid",
            "q_block_sequence",
            "q_block_start",
            "full_kv_block_sequence",
            "full_kv_block_start",
            "tail_kv_block_sequence",
            "tail_kv_block_start",
    ):
        setattr(plan, name, _H12DSHostTensor((task_capacity, )))
    state = SimpleNamespace(api=api, launches=[], launch_errors={}, validations=[], q=q, k=k, v=v, o=o, do=do,
                            lse=object(), plan=plan)
    monkeypatch.setattr(amd_fa_varlen_bwd, "torch", api)
    # Input validation has separate coverage with real tensors; these tests need
    # metadata-only inputs so an accidental huge workspace cannot reach a GPU.
    monkeypatch.setattr(amd_fa_varlen_bwd, "_validate_backward_inputs", lambda *args: state.validations.append(args))

    class Kernel:

        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):

            def launch(*args, **kwargs):
                state.launches.append((self.name, grid, args, kwargs))
                if self.name in state.launch_errors:
                    raise state.launch_errors[self.name]

            return launch

    for name in (
            "_varlen_bwd_preprocess",
            "_varlen_bwd_preprocess_dynamic_owner_queue",
            "_prefix_h12_ds_producer",
            "_prefix_h12_ds_dq_consumer",
            "_varlen_dkdv_reduce_kernel",
            "_prefix_h64_packed_ds_producer",
            "_prefix_h64_packed_ds_dq_consumer",
            "_varlen_bwd_interleaved_bm16_bn256_fp32_kernel",
            "_varlen_bwd_interleaved_bm32_kernel",
            "_varlen_bwd_interleaved_bm32_rolling_fp32_queue_s3",
            "_varlen_bwd_interleaved_bm32_rolling_fp32_queue_s4",
            "_varlen_bwd_interleaved_kernel",
            "_varlen_mha_dq_convert_coalesced_fp32_swizzled_kernel",
            "_varlen_mha_dq_convert_coalesced_kernel",
            "_varlen_dq_convert_kernel",
    ):
        monkeypatch.setattr(amd_fa_varlen_bwd, name, Kernel(name))

    def run():
        return amd_fa_varlen_bwd.fa_varlen_backward(
            q,
            k,
            v,
            o,
            do,
            state.lse,
            plan,
            128**-0.5,
            causal=case == "causal",
            dq_atomic_fp32=case != "bf16",
        )

    state.run = run
    return state


@pytest.mark.parametrize("mode", ["eager", "capture", "oom", "ineligible", "caller_capture", "same_device"])
def test_varlen_d128_h12_ds_allocation_host(mode):
    api = _H12DSHostTorch()
    q = object() if mode == "ineligible" else _H12DSHostTensor((50754, 12, 128))
    api.cuda.capturing["cuda:3"] = mode == "capture"
    api.cuda.capturing["cuda:0"] = mode == "caller_capture"
    previous = "cuda:3" if mode == "same_device" else "cuda:0"
    api.cuda.current_device = previous
    if mode == "oom":
        api.errors[((12, 51039, 10496), 1)] = api.cuda.OutOfMemoryError("optional dS")
    result = amd_fa_varlen_bwd._try_allocate_prefix_h12_ds(
        q,
        51039,
        eligible_h12=mode != "ineligible",
        torch_api=api,
    )
    assert api.cuda.current_device == previous
    assert api.cuda.streams == {"cuda:0": "caller_stream", "cuda:3": "input_stream"}
    if mode == "ineligible":
        assert result is None and not api.cuda.events and not api.allocations
        return
    assert api.cuda.events == [
        ("enter", "cuda:3"),
        ("query", "cuda:3", "input_stream"),
        ("restore", previous),
    ]
    assert len(api.allocations) == (mode != "capture")
    if api.allocations:
        kind, tensor, current_device = api.allocations[0]
        assert (kind, tensor.shape, tensor.dtype, tensor.device, current_device) == (
            "empty",
            (12, 51039, 10496),
            torch.bfloat16,
            "cuda:3",
            "cuda:3",
        )
    assert (result is None) == (mode in ("capture", "oom"))
    if result is not None:
        assert result is api.allocations[0][1]


@pytest.mark.parametrize("mode", ["eager", "capture", "optional_oom"])
@pytest.mark.parametrize("task_capacity", [400, 513])
def test_varlen_d128_h12_ds_dispatch_host(monkeypatch, mode, task_capacity):
    state = _h12_ds_host_fixture(monkeypatch, task_capacity=task_capacity)
    api = state.api
    api.cuda.capturing["cuda:3"] = mode == "capture"
    if mode == "optional_oom":
        api.errors[((12, 51039, 10496), 1)] = api.cuda.OutOfMemoryError("optional dS")
    outputs = state.run()
    handoff = mode == "eager"
    assert len(state.validations) == 1
    assert [call[0] for call in state.launches] == [
        "_varlen_bwd_preprocess",
        "_prefix_h12_ds_producer" if handoff else "_varlen_bwd_interleaved_bm16_bn256_fp32_kernel",
        *([] if handoff else ["_varlen_dkdv_reduce_kernel"]),
        "_prefix_h12_ds_dq_consumer" if handoff else "_varlen_mha_dq_convert_coalesced_fp32_swizzled_kernel",
    ]
    assert all(result is allocation[1] for result, allocation in zip(outputs, api.allocations[:3]))
    shapes = [tensor.shape for _, tensor, _ in api.allocations]
    assert shapes.count((12, 51039, 10496)) == (mode != "capture")
    assert shapes.count((12, 51039, 128)) == (not handoff)
    preprocess, core, *remaining = state.launches
    finish = remaining[-1]
    assert preprocess[3]["ZERO_DQ"] is (not handoff)
    assert preprocess[3]["PACK_STATS_MHA16"] is True
    assert preprocess[3]["PACK_STATS"] is False
    workspace_shape = (12, 51039, 10496 if handoff else 128)
    workspace, = (tensor for _, tensor, _ in api.allocations if tensor.shape == workspace_shape)
    assert preprocess[2][4] is (outputs[0] if handoff else workspace)
    assert core[2][10] is workspace
    assert core[1] == ((4, task_capacity) if handoff else (4, task_capacity, 2))
    assert core[3]["Q_SPLITS"] == (1 if handoff else 2) and core[3]["CHUNKED_Q"] is True
    assert core[3]["SORT_TASKS"] is (task_capacity <= 512)
    if handoff:
        assert core[2][11] is outputs[1] and core[2][12] is outputs[2]
        assert not any(len(shape) == 4 for shape in shapes)
    else:
        reduce = remaining[0]
        assert reduce[3]["KV_SPLITS"] == 2 and reduce[3]["FLATTEN_HEADS"] is True
        partials = [tensor for _, tensor, _ in api.allocations if tensor.shape == (100696, 4, 2, 128)]
        assert len(partials) == 2 and all(tensor.dtype is torch.float32 for tensor in partials)
        for index, tensor in enumerate(partials):
            assert core[2][11 + index] is tensor and reduce[2][index] is tensor
    assert finish[2][0] is workspace
    assert finish[2][5] is outputs[0]
    if handoff:
        assert workspace.dtype is torch.bfloat16
        assert core[3]["CuKV"] is state.plan.cu_seqlens_k and core[3]["DS_CAP"] == 10496
        assert finish[3]["enable_fp_fusion"] is False
    else:
        assert workspace.dtype is torch.float32
    assert api.cuda.current_device == "cuda:0"
    assert [event for event in api.cuda.events if event[0] == "query"] == [
        ("query", "cuda:3", "input_stream"),
    ]


@pytest.mark.parametrize("case", [
    "h64",
    "shifted_q",
    "shifted_k",
    "shifted_v",
    "shifted_do",
    "bf16",
    "causal",
    "batch",
    "total_q",
    "total_kv",
    "max_q",
    "max_kv",
])
def test_varlen_d128_h12_ds_ineligible_host(monkeypatch, case):
    state = _h12_ds_host_fixture(monkeypatch, case=case)
    # A failing query makes an accidental helper call fail visibly.
    state.api.cuda.errors["query"] = AssertionError("ineligible route queried capture")
    state.run()
    assert not state.api.cuda.events
    assert all(tensor.shape[-1] != 10496 for _, tensor, _ in state.api.allocations)
    assert all("_prefix_h12_ds_" not in call[0] for call in state.launches)
    if case == "h64":
        assert [call[0] for call in state.launches] == [
            "_varlen_bwd_preprocess",
            "_varlen_bwd_interleaved_bm16_bn256_fp32_kernel",
            "_varlen_mha_dq_convert_coalesced_fp32_swizzled_kernel",
        ]
        assert state.launches[1][3]["Q_SPLITS"] == 1


@pytest.mark.parametrize("failure", [
    "optional_runtime",
    "optional_memory",
    "optional_value",
    "fallback_after_capture",
    "fallback_after_oom",
    "dq_output",
    "dk_output",
    "dv_output",
    "dk_partial",
    "dv_partial",
    "delta",
    "query",
    "enter",
    "exit",
    "preprocess",
    "producer",
    "reducer",
    "consumer",
    "fallback_core",
])
def test_varlen_d128_h12_ds_error_propagation_host(monkeypatch, failure):
    state = _h12_ds_host_fixture(monkeypatch)
    api = state.api
    error_type = {"optional_runtime": RuntimeError, "optional_memory": MemoryError, "optional_value":
                  ValueError}.get(failure, api.cuda.OutOfMemoryError)
    error = error_type("CUDA out of memory: injected " + failure)
    if failure in ("dk_partial", "dv_partial"):
        api.cuda.capturing["cuda:3"] = True
    allocations = {
        "dq_output": ((50754, 12, 128), 1),
        "dk_output": ((100696, 4, 128), 1),
        "dv_output": ((100696, 4, 128), 2),
        "dk_partial": ((100696, 4, 2, 128), 1),
        "dv_partial": ((100696, 4, 2, 128), 2),
        "delta": ((12, 101508), 1),
    }
    if failure in allocations:
        api.errors[allocations[failure]] = error
    elif failure.startswith("optional_"):
        api.errors[((12, 51039, 10496), 1)] = error
    elif failure.startswith("fallback_after_"):
        api.cuda.capturing["cuda:3"] = failure == "fallback_after_capture"
        if failure == "fallback_after_oom":
            api.errors[((12, 51039, 10496), 1)] = api.cuda.OutOfMemoryError("optional dS")
        api.errors[((12, 51039, 128), 1)] = error
    elif failure in ("query", "enter", "exit"):
        api.cuda.errors[failure] = error
    else:
        kernel = {
            "preprocess": "_varlen_bwd_preprocess",
            "producer": "_prefix_h12_ds_producer",
            "reducer": "_varlen_dkdv_reduce_kernel",
            "consumer": "_prefix_h12_ds_dq_consumer",
            "fallback_core": "_varlen_bwd_interleaved_bm16_bn256_fp32_kernel",
        }[failure]
        api.cuda.capturing["cuda:3"] = failure in ("fallback_core", "reducer")
        state.launch_errors[kernel] = error
    with pytest.raises(error_type) as caught:
        state.run()
    assert caught.value is error
    assert api.cuda.current_device == "cuda:0"
    assert api.cuda.streams == {"cuda:0": "caller_stream", "cuda:3": "input_stream"}
    shapes = [tensor.shape for _, tensor, _ in api.allocations]
    assert shapes.count((12, 51039, 10496)) <= 1
    assert shapes.count((12, 51039, 128)) == (failure.startswith("fallback_after_")
                                              or failure in ("fallback_core", "reducer", "dk_partial", "dv_partial"))
    if state.launch_errors:
        assert state.launches[-1][0] == kernel
        assert sum(call[0] == kernel for call in state.launches) == 1
    else:
        assert not state.launches
    if failure in ("dk_partial", "dv_partial"):
        assert api.cuda.events == [
            ("enter", "cuda:3"),
            ("query", "cuda:3", "input_stream"),
            ("restore", "cuda:0"),
        ]
    elif failure in allocations:
        assert not api.cuda.events


_H64_DS_ROLES = (
    "_varlen_bwd_preprocess",
    "_prefix_h64_packed_ds_producer",
    "_prefix_h64_packed_ds_dq_consumer",
    "_varlen_bwd_interleaved_bm16_bn256_fp32_kernel",
    "_varlen_mha_dq_convert_coalesced_fp32_swizzled_kernel",
    "_varlen_dkdv_reduce_kernel",
)


def _packed_ds_offsets(q_lengths, kv_lengths):
    rectangles = (((q + 15) // 16) * 16 * ((k + 255) // 256) * 256 for q, k in zip(q_lengths, kv_lengths, strict=True))
    return [0, *itertools.accumulate(rectangles)]


def _assert_h64_ds_route(records, handoff):
    expected = _H64_DS_ROLES[:3] if handoff else (_H64_DS_ROLES[0], _H64_DS_ROLES[3], _H64_DS_ROLES[4])
    assert [name for name, _, _ in records] == list(expected)
    preprocess = records[0][1]
    assert preprocess["ZERO_DQ"] is (not handoff)
    assert preprocess["PACK_STATS_MHA16"] is True and preprocess["PACK_STATS"] is False
    assert records[1][1]["Q_SPLITS"] == 1
    _assert_dk_native_column_subtiles(records[1][2])
    if handoff:
        assert records[-1][1]["enable_fp_fusion"] is False
        for name, _, compiled in records[1:]:
            assert "buffer_atomic" not in compiled.asm["amdgcn"], name
            _assert_scratch_free(name, compiled)
    else:
        assert "buffer_atomic_add_f32" in records[1][2].asm["amdgcn"]


@_H12_DS_METADATA
@_H12_DS_SCALES
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_h64_ds_public_gfx950(monkeypatch, supply_metadata, sm_scale):
    inputs, expected, qs, ks = _make_prefix_ds_case(supply_metadata, sm_scale, q_heads=64, kv_heads=8)
    plan = inputs[-1]
    if supply_metadata:
        assert plan.packed_ds_prefix is None and plan.packed_ds_head_elements is None
    else:
        offsets = _packed_ds_offsets(_H12_DS_Q_LENGTHS, _H12_DS_KV_LENGTHS)
        assert plan.packed_ds_prefix.dtype is torch.int64
        assert plan.packed_ds_prefix.device == inputs[0].device
        assert plan.packed_ds_prefix.tolist() == offsets
        assert plan.packed_ds_head_elements == offsets[-1]
    records = _observe_prefix_ds_kernels(monkeypatch, roles=_H64_DS_ROLES)
    actual = amd_fa_varlen_bwd.fa_varlen_backward(*inputs, sm_scale, dq_atomic_fp32=True)
    _assert_h64_ds_route(records, not supply_metadata)
    _assert_prefix_ds_result(actual, inputs, expected, qs, ks)


@pytest.mark.parametrize("mode", ("old_plan", "oom"))
@_H12_DS_SCALES
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_h64_ds_fallback_gfx950(monkeypatch, mode, sm_scale):
    inputs, expected, qs, ks = _make_prefix_ds_case(False, sm_scale, q_heads=64, kv_heads=8)
    plan = inputs[-1]
    empty = torch.empty
    attempts = []

    def injected_empty(*args, **kwargs):
        shape = tuple(args[0]) if args and isinstance(args[0], (tuple, list)) else None
        if shape == (64, plan.packed_ds_head_elements) and kwargs.get("dtype") is torch.bfloat16:
            attempts.append(shape)
            raise torch.cuda.OutOfMemoryError("optional compact dS")
        return empty(*args, **kwargs)

    if mode == "old_plan":
        inputs = (*inputs[:-1], replace(plan, packed_ds_prefix=None, packed_ds_head_elements=None))
    monkeypatch.setattr(torch, "empty", injected_empty)
    records = _observe_prefix_ds_kernels(monkeypatch, roles=_H64_DS_ROLES)
    actual = amd_fa_varlen_bwd.fa_varlen_backward(*inputs, sm_scale, dq_atomic_fp32=True)
    _assert_h64_ds_route(records, False)
    _assert_prefix_ds_result(actual, inputs, expected, qs, ks)
    assert len(attempts) == (mode == "oom")


@pytest.mark.parametrize("fixture", ("short", "chunks"))
@pytest.mark.parametrize("sorted_tasks", (False, True))
@_H12_DS_SCALES
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_h64_ds_producer_boundaries_gfx950(fixture, sorted_tasks, sm_scale):
    if fixture == "short":
        qs = [1, 15, 16, 17, 31, 32, 63, 64, 127, 128, 129]
        ks = [1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 513]
    else:
        qs, ks = [511, 512, 513, 1025], [255, 256, 257, 1281]
    case = _make_varlen_d128_reference_case(qs, ks, q_heads=64, kv_heads=8, seed=9417, sm_scale=sm_scale)
    q, k, v, out, do, lse, cu_q, cu_kv, _, expected = case
    if fixture == "short":
        # With one key, P=1 and dQ=dK=0. Use exact binary fractions so
        # dp and delta both equal 128 * (1/8)**2; relative error against
        # a cancellation residual would otherwise be ill-conditioned.
        for tensor in (v, out, do):
            tensor[0].fill_(0.125)
        expected[0][0].zero_()
        expected[1][0].zero_()
        expected[2][0].fill_(1.0)  # Eight sibling heads each contribute 1/8.
    plan = amd_fa_varlen_bwd.prepare_varlen_backward(cu_q, cu_kv, sum(qs), sum(ks), max(qs), max(ks))
    offsets = _packed_ds_offsets(qs, ks)
    prefix = torch.tensor(offsets, dtype=torch.int64, device="cuda")
    # Check both ends of the workspace and the final dQ output. Poisoning the
    # interior also exposes uninitialized Q16 padding or missing KV256 stores.
    ds_storage = torch.full((64 * offsets[-1] + 8192, ), float("nan"), dtype=torch.bfloat16, device="cuda")
    ds = ds_storage[4096:-4096].view(64, offsets[-1])
    dq_storage = torch.full((q.shape[0] + 128, 64, 128), float("nan"), dtype=torch.bfloat16, device="cuda")
    dq = dq_storage[:q.shape[0]]
    dk, dv = torch.full_like(k, float("nan")), torch.full_like(v, float("nan"))
    delta = torch.empty((64, 2 * q.shape[0]), dtype=torch.float32, device="cuda")
    padded = q.shape[0] + 15 * len(qs)
    amd_fa_varlen_bwd._varlen_bwd_preprocess[((max(qs) + 63) // 64,
                                              len(qs) * 64)](out, do, delta, cu_q, dq, padded, plan.task_counts,
                                                             PLAN_ERROR_INDEX=4, HEADS=64, D=128, BLOCK_M=64,
                                                             ZERO_DQ=False, DQ_PAD_ROWS=16, LSE=lse, TOTAL_Q=q.shape[0],
                                                             PACK_STATS=False, PACK_STATS_MHA16=True, num_warps=4)
    producer = amd_fa_varlen_bwd._prefix_h64_packed_ds_producer[(8, plan.wide_kv_start.numel())](
        q, k, v, do, delta, plan.wide_kv_start, plan.wide_q_start, plan.wide_dq_start, plan.wide_q_len,
        plan.wide_kv_valid, ds, dk, dv, plan.task_counts, CuKV=cu_kv, DS_PREFIX=prefix, DS_HEAD_ELEMENTS=offsets[-1],
        SM_SCALE=sm_scale, TOTAL_Q=q.shape[0], TOTAL_Q_PADDED=padded, HQ=64, HKV=8, D=128, BLOCK_M=16, BLOCK_N=256,
        TASK_COUNT_INDEX=3, PLAN_ERROR_INDEX=4, CHUNKED_Q=True, Q_SPLITS=1, SORT_TASKS=sorted_tasks, DS_CAP=10496,
        num_warps=4, num_stages=1, matrix_instr_nonkdim=16, reverse_local_assignment=True,
        enable_sched_group_barrier_scheduler=False, llvm_fn_attrs=())
    for seq, (ql, kl) in enumerate(zip(qs, ks, strict=True)):
        block = ds[:, offsets[seq]:offsets[seq + 1]].reshape(64, (ql + 15) // 16, ((kl + 255) // 256) * 256, 16)
        assert torch.isfinite(block).all()
        assert torch.count_nonzero(block[:, -1, :, (ql % 16 or 16):]).item() == 0
        assert torch.count_nonzero(block[:, :, kl:, :]).item() == 0
    consumer = amd_fa_varlen_bwd._prefix_h64_packed_ds_dq_consumer[((max(qs) + 127) // 128, 64, len(qs))](
        ds, k, cu_q, cu_kv, plan.task_counts, dq, TOTAL_Q_PADDED=padded, DS_PREFIX=prefix, DS_HEAD_ELEMENTS=offsets[-1],
        SM_SCALE=sm_scale, DS_CAP=10496, PLAN_ERROR_INDEX=4, num_warps=4, matrix_instr_nonkdim=16,
        enable_fp_fusion=False)
    if fixture == "short":
        assert torch.count_nonzero(dq[0]).item() == 0
        assert torch.count_nonzero(dk[0]).item() == 0
        assert torch.equal(dv[0], torch.ones_like(dv[0]))
    _assert_prefix_ds_result((dq, dk, dv), (q, k, v), expected, qs, ks)
    assert torch.isnan(ds_storage[:4096]).all() and torch.isnan(ds_storage[-4096:]).all()
    assert torch.isnan(dq_storage[q.shape[0]:]).all()
    _assert_scratch_free("producer", producer)
    _assert_dk_native_column_subtiles(producer)
    _assert_scratch_free("consumer", consumer)


@_H12_DS_SCALES
@pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950 hardware")
def test_varlen_d128_h64_ds_consumer_boundaries_gfx950(sm_scale):
    qs = (127, 128, 129, 511, 512, 513, 1, 17, 33)
    ks = (1, 63, 64, 65, 255, 256, 257, 513, 0)
    qo, ko = [0, *itertools.accumulate(qs)], [0, *itertools.accumulate(ks)]
    offsets = _packed_ds_offsets(qs, ks)
    ds_storage = torch.full((64 * offsets[-1] + 8192, ), float("nan"), dtype=torch.bfloat16, device="cuda")
    ds = ds_storage[4096:-4096].view(64, offsets[-1])
    k = torch.full((ko[-1] + 64, 8, 128), float("nan"), dtype=torch.bfloat16, device="cuda")
    dq = torch.full((qo[-1] + 128, 64, 128), float("nan"), dtype=torch.bfloat16, device="cuda")
    expected = torch.zeros_like(dq[:qo[-1]])
    # Exercise several siblings and the last query/KV head. Padding stays NaN
    # outside each initialized rectangle; the empty-KV sequence must write zero.
    for seq, (ql, kl) in enumerate(zip(qs, ks, strict=True)):
        block = ds[:, offsets[seq]:offsets[seq + 1]].reshape(64, (ql + 15) // 16, ((kl + 255) // 256) * 256, 16)
        block.zero_()
        head = (0, 1, 2, 3, 5, 7, 8, 63, 10)[seq]
        for m in range((ql + 15) // 16):
            block[head, m, :kl, :min(16, ql - 16 * m)].fill_(0.25)
        for kh in range(8):
            k[ko[seq]:ko[seq + 1], kh].fill_(kh + 1)
        total = torch.zeros((), dtype=torch.float32, device="cuda")
        for n in range(0, kl, 256):
            partial = torch.tensor(min(256, kl - n) * 0.25 * (head // 8 + 1), dtype=torch.float32, device="cuda")
            total = total + partial * sm_scale
        expected[qo[seq]:qo[seq + 1], head].fill_(total.to(torch.bfloat16))
    cu_q = torch.tensor(qo, dtype=torch.int32, device="cuda")
    cu_kv = torch.tensor(ko, dtype=torch.int32, device="cuda")
    prefix = torch.tensor(offsets, dtype=torch.int64, device="cuda")
    counts = torch.zeros(6, dtype=torch.int32, device="cuda")
    amd_fa_varlen_bwd._prefix_h64_packed_ds_dq_consumer[((max(qs) + 127) // 128, 64, len(qs))](
        ds, k, cu_q, cu_kv, counts, dq, TOTAL_Q_PADDED=qo[-1] + 15 * len(qs), DS_PREFIX=prefix,
        DS_HEAD_ELEMENTS=offsets[-1], SM_SCALE=sm_scale, DS_CAP=10496, PLAN_ERROR_INDEX=4, num_warps=4,
        matrix_instr_nonkdim=16, enable_fp_fusion=False)
    assert torch.equal(dq[:qo[-1]], expected)
    assert torch.isnan(dq[qo[-1]:]).all() and torch.isnan(k[ko[-1]:]).all()
    assert torch.isnan(ds_storage[:4096]).all() and torch.isnan(ds_storage[-4096:]).all()


@pytest.mark.parametrize("mode", ("eager", "capture", "oom", "missing_metadata", "caller_capture"))
@pytest.mark.parametrize("task_capacity", (403, 513))
def test_varlen_d128_h64_ds_dispatch_host(monkeypatch, mode, task_capacity):
    state = _h12_ds_host_fixture(monkeypatch, case="h64", task_capacity=task_capacity)
    # The real plan preparation is covered by the public GPU test; these
    # metadata-only tensors keep allocation and fallback tests off the GPU.
    elements = 306958336
    if mode != "missing_metadata":
        state.plan.packed_ds_prefix = object()
        state.plan.packed_ds_head_elements = elements
    state.api.cuda.capturing["cuda:3"] = mode == "capture"
    state.api.cuda.capturing["cuda:0"] = mode == "caller_capture"
    if mode == "oom":
        state.api.errors[((64, elements), 1)] = state.api.cuda.OutOfMemoryError("optional compact dS")
    outputs = state.run()
    handoff = mode in ("eager", "caller_capture")
    expected = _H64_DS_ROLES[:3] if handoff else (_H64_DS_ROLES[0], _H64_DS_ROLES[3], _H64_DS_ROLES[4])
    assert [call[0] for call in state.launches] == list(expected)
    preprocess, producer, consumer = state.launches
    assert preprocess[3]["ZERO_DQ"] is (not handoff)
    assert preprocess[3]["PACK_STATS_MHA16"] is True and preprocess[3]["PACK_STATS"] is False
    assert producer[3]["Q_SPLITS"] == 1 and producer[3]["SORT_TASKS"] is (task_capacity <= 512)
    assert producer[2][11] is outputs[1] and producer[2][12] is outputs[2]
    assert outputs[1].dtype is outputs[2].dtype is torch.bfloat16
    shapes = [tensor.shape for _, tensor, _ in state.api.allocations]
    assert shapes.count((64, elements)) == (mode not in ("capture", "missing_metadata"))
    assert shapes.count((64, 51039, 128)) == (not handoff)
    assert not any(len(shape) == 4 for shape in shapes)
    if handoff:
        assert producer[3]["DS_PREFIX"] is consumer[3]["DS_PREFIX"] is state.plan.packed_ds_prefix
        assert producer[3]["DS_HEAD_ELEMENTS"] == consumer[3]["DS_HEAD_ELEMENTS"] == elements
        assert consumer[3]["enable_fp_fusion"] is False
    assert state.api.cuda.current_device == "cuda:0"
    assert state.api.cuda.streams == {"cuda:0": "caller_stream", "cuda:3": "input_stream"}
    queries = [event for event in state.api.cuda.events if event[0] == "query"]
    assert queries == ([] if mode == "missing_metadata" else [("query", "cuda:3", "input_stream")])


@pytest.mark.parametrize("failure", (
    "optional_runtime",
    "optional_memory",
    "optional_value",
    "preprocess",
    "producer",
    "consumer",
    "query",
    "enter",
    "exit",
    "fallback_after_capture",
    "fallback_after_oom",
))
def test_varlen_d128_h64_ds_error_propagation_host(monkeypatch, failure):
    state = _h12_ds_host_fixture(monkeypatch, case="h64")
    elements = 306958336
    state.plan.packed_ds_prefix = object()
    state.plan.packed_ds_head_elements = elements
    api = state.api
    error_type = {"optional_runtime": RuntimeError, "optional_memory": MemoryError, "optional_value":
                  ValueError}.get(failure, api.cuda.OutOfMemoryError)
    error = error_type("injected compact dS failure")
    if failure.startswith("optional_"):
        api.errors[((64, elements), 1)] = error
    elif failure.startswith("fallback_after_"):
        api.cuda.capturing["cuda:3"] = failure == "fallback_after_capture"
        if failure == "fallback_after_oom":
            api.errors[((64, elements), 1)] = api.cuda.OutOfMemoryError("optional compact dS")
        api.errors[((64, 51039, 128), 1)] = error
    elif failure in ("query", "enter", "exit"):
        api.cuda.errors[failure] = error
    else:
        kernel = dict(zip(("preprocess", "producer", "consumer"), _H64_DS_ROLES[:3], strict=True))[failure]
        state.launch_errors[kernel] = error
    with pytest.raises(error_type) as caught:
        state.run()
    assert caught.value is error
    assert api.cuda.current_device == "cuda:0"
    assert api.cuda.streams == {"cuda:0": "caller_stream", "cuda:3": "input_stream"}
    shapes = [tensor.shape for _, tensor, _ in api.allocations]
    assert shapes.count((64, elements)) <= 1
    assert shapes.count((64, 51039, 128)) == failure.startswith("fallback_after_")
