# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Parallel softmax backward schedules for small gfx950 batches.

The private launch cache preserves ordinary JIT specialization and runtime hooks.
Clear both caches after changing compiler options outside the runtime option key.
"""

from collections import OrderedDict
import inspect
from threading import Lock

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.runtime.jit import JITFunction
from tlx_gfx950_cross_attention_v3_baseline import _tlx_gfx950_cross_attn_v3_ttgir_bwd


def _scalar_key(value):
    if value is None or type(value) in (bool, int, str):
        return (type(value), value)
    if type(value) is float:
        return (float, value.hex())
    if type(value) in (tuple, list):
        return (type(value), tuple(_scalar_key(item) for item in value))
    if type(value) is dict:
        return (dict, tuple(sorted((key, _scalar_key(item)) for key, item in value.items())))
    raise TypeError('Unsupported specialization value')


def _argument_key(value):
    if type(value) is torch.Tensor:
        if value.layout != torch.strided or value.device.type != 'cuda':
            raise TypeError('Unsupported tensor layout or device')
        return (torch.Tensor, value.dtype, value.device, value.data_ptr() % 16 == 0, value.untyped_storage().nbytes()
                <= 2**31 - 1)
    return _scalar_key(value)


class _GuardedLauncher:

    def __init__(self, function, max_entries=64, bound_options=None):
        if not isinstance(function, JITFunction):
            raise TypeError('Only plain JITFunction objects are supported')
        self.function = function
        self.max_entries = max_entries
        self.names = tuple(function.signature.parameters)
        self.parameter_names = frozenset(self.names)
        self.bound_options = bound_options
        self.defaults = {
            name: parameter.default
            for name, parameter in function.signature.parameters.items()
            if parameter.default is not inspect.Parameter.empty
        }
        self.entries = OrderedDict()
        self.lock = Lock()
        self.hits = self.misses = self.fallbacks = self.evictions = 0

    def __getitem__(self, grid):
        if callable(grid):
            return self.function[grid]
        grid = tuple(grid) + (1, ) * (3 - len(grid))

        def launch(*args, **kwargs):
            return self.launch(grid, args, kwargs)

        return launch

    def _semantic_hooks(self):
        runtime = triton.knobs.runtime
        return (self.function.pre_run_hooks or runtime.jit_cache_hook or runtime.jit_post_compile_hook
                or runtime.add_stages_inspection_hook or triton.knobs.compilation.always_compile
                or triton.knobs.compilation.listener)

    def _key(self, grid, complete, options):
        runtime = triton.knobs.runtime
        function = self.function
        # Reading cache_key also discovers source dependencies before lookup.
        source_key = function.cache_key
        for (name, _), (expected, namespace) in function.used_global_vals.items():
            if namespace.get(name) != expected:
                raise TypeError('A source global changed')
        runtime_key = (runtime.debug, runtime.sanitize_overflow, triton.knobs.compilation.instrumentation_mode,
                       triton.knobs.compilation.fpsan_homomorphic_casts)
        keys = self._argument_keys(complete)
        return (source_key, function.src, function.debug, function.noinline, grid, torch.cuda.current_device(), keys,
                _scalar_key(options), runtime_key)

    def _argument_keys(self, complete):
        tensor_keys = {}
        keys = []
        for value in complete:
            if type(value) is torch.Tensor:
                identity = id(value)
                value_key = tensor_keys.get(identity)
                if value_key is None:
                    value_key = _argument_key(value)
                    tensor_keys[identity] = value_key
            else:
                value_key = _scalar_key(value)
            keys.append(value_key)
        return tuple(keys)

    def launch(self, grid, args, kwargs):
        function = self.function
        if self._semantic_hooks():
            self.fallbacks += 1
            return function[grid](*args, **kwargs)
        try:
            if (kwargs is self.bound_options and len(args) == len(self.names)
                    and self.parameter_names.isdisjoint(kwargs)):
                complete = args
                options = kwargs
            else:
                if len(args) > len(self.names):
                    raise TypeError('Unexpected argument count')
                if any(name in kwargs for name in self.names[:len(args)]):
                    raise TypeError('Duplicate argument')
                complete = args + tuple(kwargs[name] if name in kwargs else self.defaults[name]
                                        for name in self.names[len(args):])
                options = {name: value for name, value in kwargs.items() if name not in self.names}
            key = self._key(grid, complete, options)
            with self.lock:
                entry = self.entries.get(key)
                if entry is not None:
                    kernel, runner, device_cache, jit_key = entry
                    if (function.device_caches.get(torch.cuda.current_device()) is not device_cache
                            or device_cache[0].get(jit_key) is not kernel):
                        self.entries.pop(key, None)
                        entry = None
                if entry is not None:
                    self.entries.move_to_end(key)
                    self.hits += 1
        except (TypeError, KeyError, ValueError):
            self.fallbacks += 1
            return function[grid](*args, **kwargs)
        if entry is None:
            self.misses += 1
            kernel = function[grid](*args, **kwargs)
            if kernel is not None:
                device_cache = function.device_caches.get(torch.cuda.current_device())
                if device_cache is not None:
                    jit_key = next((name for name, compiled in tuple(device_cache[0].items()) if compiled is kernel),
                                   None)
                    if jit_key is not None:
                        entry = (kernel, kernel[grid], device_cache, jit_key)
                        with self.lock:
                            self.entries[key] = entry
                            if len(self.entries) > self.max_entries:
                                self.entries.popitem(last=False)
                                self.evictions += 1
            return kernel
        kernel, runner, _, _ = entry
        runner(*complete)
        return kernel

    def clear(self):
        with self.lock:
            self.entries.clear()

    def counters(self):
        with self.lock:
            return dict(hits=self.hits, misses=self.misses, fallbacks=self.fallbacks, evictions=self.evictions,
                        entries=len(self.entries))


@triton.jit
def _small_bwd_impl(Q, K, V, DO, M, OUT, SOQ, SOK, DQ, DK, DV, alpha, H: tl.constexpr, KV_TILES: tl.constexpr,
                    SHARED: tl.constexpr, KV_FIRST: tl.constexpr = False):
    BLOCK: tl.constexpr = 64
    HEAD_DIM: tl.constexpr = 128
    QUERY_TILES: tl.constexpr = 4
    head_batch = tl.program_id(0)
    task = tl.program_id(1)
    if KV_FIRST:
        task = (task + QUERY_TILES) % (QUERY_TILES + KV_TILES)
    batch = head_batch // H
    head = head_batch % H
    d = tl.arange(0, HEAD_DIM)
    q_start = tl.load(SOQ + batch).to(tl.int64)
    q_end = tl.load(SOQ + batch + 1).to(tl.int64)
    k_start = tl.load(SOK + batch).to(tl.int64)
    k_end = tl.load(SOK + batch + 1).to(tl.int64)

    if task < QUERY_TILES:
        if k_end - k_start != 1:
            rows = q_start + task * BLOCK + tl.arange(0, BLOCK)
            q_mask = rows < q_end
            q_offsets = (rows[:, None] * H + head) * HEAD_DIM + d[None, :]
            q = tl.load(Q + q_offsets, mask=q_mask[:, None], other=0)
            do = tl.load(DO + q_offsets, mask=q_mask[:, None], other=0)
            out = tl.load(OUT + q_offsets, mask=q_mask[:, None], other=0)
            m = tl.load(M + rows * H + head, mask=q_mask, other=0)
            delta = tl.sum(out.to(tl.float32) * do.to(tl.float32), axis=1)
            dq = tl.full((BLOCK, HEAD_DIM), 0, tl.float32)
            for tile in range(KV_TILES):
                columns = k_start + tile * BLOCK + tl.arange(0, BLOCK)
                k_mask = columns < k_end
                k_offsets = (columns[:, None] * H + head) * HEAD_DIM + d[None, :]
                k = tl.load(K + k_offsets, mask=k_mask[:, None], other=0)
                if SHARED:
                    v = k
                else:
                    v = tl.load(V + k_offsets, mask=k_mask[:, None], other=0)
                scores = tl.dot(q, tl.trans(k), allow_tf32=False)
                p = tl.exp2(scores * (alpha * 1.44269504) - m[:, None])
                dp = tl.dot(do, tl.trans(v), allow_tf32=False)
                ds = tl.where(q_mask[:, None] & k_mask[None, :], p * (dp - delta[:, None]), 0)
                dq = tl.dot(ds.to(tl.bfloat16), k, dq, allow_tf32=False)
            tl.store(DQ + q_offsets, dq * alpha, mask=q_mask[:, None])
        else:
            rows = q_start + task * BLOCK + tl.arange(0, BLOCK)
            offsets = (rows[:, None] * H + head) * HEAD_DIM + d[None, :]
            tl.store(DQ + offsets, 0.0, mask=(rows < q_end)[:, None])
    else:
        if k_end - k_start != 1:
            columns = k_start + (task - QUERY_TILES) * BLOCK + tl.arange(0, BLOCK)
            k_mask = columns < k_end
            k_offsets = (columns[:, None] * H + head) * HEAD_DIM + d[None, :]
            k = tl.load(K + k_offsets, mask=k_mask[:, None], other=0)
            if SHARED:
                v = k
            else:
                v = tl.load(V + k_offsets, mask=k_mask[:, None], other=0)
            dk = tl.full((BLOCK, HEAD_DIM), 0, tl.float32)
            dv = tl.full((BLOCK, HEAD_DIM), 0, tl.float32)
            for tile in range(QUERY_TILES):
                rows = q_start + tile * BLOCK + tl.arange(0, BLOCK)
                q_mask = rows < q_end
                q_offsets = (rows[:, None] * H + head) * HEAD_DIM + d[None, :]
                q = tl.load(Q + q_offsets, mask=q_mask[:, None], other=0)
                do = tl.load(DO + q_offsets, mask=q_mask[:, None], other=0)
                out = tl.load(OUT + q_offsets, mask=q_mask[:, None], other=0)
                m = tl.load(M + rows * H + head, mask=q_mask, other=0)
                delta = tl.sum(out.to(tl.float32) * do.to(tl.float32), axis=1)
                valid = k_mask[:, None] & q_mask[None, :]
                scores = tl.dot(k, tl.trans(q), allow_tf32=False)
                p = tl.exp2(scores * (alpha * 1.44269504) - m[None, :])
                p = tl.where(valid, p, 0)
                dp = tl.dot(v, tl.trans(do), allow_tf32=False)
                ds = tl.where(valid, p * (dp - delta[None, :]), 0)
                dv = tl.dot(p.to(tl.bfloat16), do, dv, allow_tf32=False)
                dk = tl.dot(ds.to(tl.bfloat16), q, dk, allow_tf32=False)
            dk = dk * alpha
            if SHARED:
                dk += dv
            tl.store(DK + k_offsets, dk, mask=k_mask[:, None])
            if not SHARED:
                tl.store(DV + k_offsets, dv, mask=k_mask[:, None])
        elif task == QUERY_TILES:
            value_gradient = tl.full((HEAD_DIM, ), 0, tl.float32)
            for tile in range(QUERY_TILES):
                rows = q_start + tile * BLOCK + tl.arange(0, BLOCK)
                offsets = (rows[:, None] * H + head) * HEAD_DIM + d[None, :]
                values = tl.load(DO + offsets, mask=(rows < q_end)[:, None], other=0).to(tl.float32)
                value_gradient += tl.sum(values, axis=0)
            offsets = (k_start * H + head) * HEAD_DIM + d
            if SHARED:
                tl.store(DK + offsets, value_gradient)
            else:
                tl.store(DK + offsets, 0.0)
                tl.store(DV + offsets, value_gradient)


@triton.jit
def _small_bwd_shared(Q, K, DO, M, OUT, SOQ, SOK, DQ, DK, alpha, H: tl.constexpr, KV_TILES: tl.constexpr):
    _small_bwd_impl(Q, K, K, DO, M, OUT, SOQ, SOK, DQ, DK, DK, alpha, H, KV_TILES, True)


@triton.jit
def _small_bwd_shared_kv_first(Q, K, DO, M, OUT, SOQ, SOK, DQ, DK, alpha, H: tl.constexpr, KV_TILES: tl.constexpr):
    _small_bwd_impl(Q, K, K, DO, M, OUT, SOQ, SOK, DQ, DK, DK, alpha, H, KV_TILES, True, True)


@triton.jit
def _small_bwd_separate(Q, K, V, DO, M, OUT, SOQ, SOK, DQ, DK, DV, alpha, H: tl.constexpr, KV_TILES: tl.constexpr):
    _small_bwd_impl(Q, K, V, DO, M, OUT, SOQ, SOK, DQ, DK, DV, alpha, H, KV_TILES, False)


_SMALL_OPTIONS = dict(num_warps=4, num_stages=1, matrix_instr_nonkdim=16, waves_per_eu=1)
_shared_launcher = _GuardedLauncher(_small_bwd_shared, bound_options=_SMALL_OPTIONS)
_shared_kv_first_launcher = _GuardedLauncher(_small_bwd_shared_kv_first, bound_options=_SMALL_OPTIONS)
_separate_launcher = _GuardedLauncher(_small_bwd_separate, bound_options=_SMALL_OPTIONS)


def small_softmax_backward(q, k, v, dout, m, out, offsets_q, offsets_kv, max_kv, alpha, shared_kv):
    """Compute gradients for contiguous BF16 inputs with Q <= 256 and D = 128."""
    dout = dout.contiguous()
    heads = q.shape[1]
    kv_tiles = triton.cdiv(max_kv, 64)
    grid = ((offsets_q.numel() - 1) * heads, 4 + kv_tiles, 1)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    if shared_kv:
        args = (q, k, dout, m, out, offsets_q, offsets_kv, dq, dk, alpha, heads, kv_tiles)
        launcher = _shared_kv_first_launcher if heads == 1 and grid[0] == 64 and max_kv == 128 else _shared_launcher
        launcher.launch(grid, args, _SMALL_OPTIONS)
        dv = None
    else:
        dv = torch.empty_like(v)
        args = (q, k, v, dout, m, out, offsets_q, offsets_kv, dq, dk, dv, alpha, heads, kv_tiles)
        _separate_launcher.launch(grid, args, _SMALL_OPTIONS)
    return dq, dk, dv


@triton.jit
def _coarse_bwd(Q, K, DO, SOQ, SOK, DQ, DK, M, OUT, TQ, SQ, SK, SDO, SDQ, SDK, alpha):
    _tlx_gfx950_cross_attn_v3_ttgir_bwd(Q, K, K, DO, TQ, 0, SOK, SOQ, DQ, DK, DK, SQ, 128, SK, 128, SK, 128, SDO, 128,
                                        SDQ, 128, SDK, 128, 128, 128, alpha, 0, 0, M, 0, 0, 0, 0, 0, 0,
                                        DIRECT_FIRST_DQ_STORE=True, ATOMIC_DQ=False, PREFETCH_DQ=True, STAGE_QDO=True,
                                        PIPELINE_QDO=True, DQ_FINAL=None, OUT=OUT, KV_SPLITS=4)


@triton.jit
def _coarse_bwd_cached(Q, K, DO, SOQ, SOK, DQ, DK, M, OUT, TQ, SQ, SK, SDO, SDQ, SDK, alpha):
    _tlx_gfx950_cross_attn_v3_ttgir_bwd(Q, K, K, DO, TQ, 0, SOK, SOQ, DQ, DK, DK, SQ, 128, SK, 128, SK, 128, SDO, 128,
                                        SDQ, 128, SDK, 128, 128, 128, alpha, 0, 0, M, 0, 0, 0, 0, 0, 0,
                                        DIRECT_FIRST_DQ_STORE=True, ATOMIC_DQ=False, PREFETCH_DQ=True, STAGE_QDO=True,
                                        PIPELINE_QDO=True, DQ_FINAL=None, OUT=OUT, KV_SPLITS=4, CACHE_DELTA=True)


@triton.jit
def _reduce_partial_dq(PARTIAL, DQ, N, DO, DK, SOQ, SOK):
    batch = tl.program_id(0)
    tile = tl.program_id(1)
    qs = tl.load(SOQ + batch).to(tl.int64)
    qe = tl.load(SOQ + batch + 1).to(tl.int64)
    ks = tl.load(SOK + batch).to(tl.int64)
    ke = tl.load(SOK + batch + 1).to(tl.int64)
    offsets = qs * 128 + tile * 2048 + tl.arange(0, 2048)
    mask = offsets < qe * 128
    if ke - ks != 1:
        value = tl.full((2048, ), 0, tl.float32)
        for split in tl.static_range(4):
            value += tl.load(PARTIAL + split * N.to(tl.int64) + offsets, mask, other=0)
        tl.store(DQ + offsets, value, mask)
    else:
        tl.store(DQ + offsets, 0., mask)
        if tile == 0:
            d = tl.arange(0, 128)
            value = tl.full((128, ), 0., tl.float32)
            for start in range(0, qe - qs, 32):
                rows = qs + start + tl.arange(0, 32)
                do = tl.load(DO + rows[:, None] * 128 + d[None, :], rows[:, None] < qe, 0)
                value += tl.sum(do.to(tl.float32), 0)
            tl.store(DK + ks * 128 + d, value)


_coarse_launcher = _GuardedLauncher(_coarse_bwd)
_coarse_cached_launcher = _GuardedLauncher(_coarse_bwd_cached)
_reduce_launcher = _GuardedLauncher(_reduce_partial_dq)
_COARSE_OPTIONS = dict(_SMALL_OPTIONS, regclass_priority_trumps_globalness=False)


def coarse_softmax_backward(q, k, dout, m, out, offsets_q, offsets_kv, alpha, cache_delta=False):
    """Split shared-K/V sequences into four disjoint blocks of KV tiles."""
    dout = dout.contiguous()
    tokens = q.shape[0]
    partial = torch.empty((4, tokens, 1, 128), device=q.device, dtype=torch.float32)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    grid = (offsets_q.numel() - 1, 4, 1)
    args = (q, k, dout, offsets_q, offsets_kv, partial, dk, m, out, tokens, q.stride(0), k.stride(0), dout.stride(0),
            128, 128, alpha)
    if cache_delta:
        launcher = _coarse_cached_launcher
    else:
        launcher = _coarse_launcher
    launcher.launch(grid, args, _COARSE_OPTIONS)
    _reduce_launcher.launch((offsets_q.numel() - 1, 16, 1), (partial, dq, q.numel(), dout, dk, offsets_q, offsets_kv),
                            {})
    return dq, dk, None


# Separate helpers keep the Q16 and KV32 tensor layouts independent.
@triton.jit
def _compact_q_owner_staged(Q, K, DO, M, OUT, DQ, qs, qe, ks, ke, head, task, alpha, H: tl.constexpr,
                            MAX_KV: tl.constexpr, BQ: tl.constexpr, KL: tl.constexpr):
    rows = qs + task * BQ + tl.arange(0, BQ)
    d = tl.arange(0, 128)
    qm = rows < qe
    qo = (rows[:, None] * H + head) * 128 + d[None, :]
    q = tl.load(Q + qo, qm[:, None], other=0)
    do = tl.load(DO + qo, qm[:, None], other=0)
    out = tl.load(OUT + qo, qm[:, None], other=0)
    m = tl.load(M + rows * H + head, qm, other=0)
    delta = tl.sum(out.to(tl.float32) * do.to(tl.float32), 1)
    # Distribute score columns across all selected warps.
    score_layout: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                     warps_per_cta=([1, 8] if H == 2 and MAX_KV == 1024 else [1, 4]))
    lhs_layout: tl.constexpr = tlx.dot_operand_layout(0, score_layout, k_width=4)
    rhs_layout: tl.constexpr = tlx.dot_operand_layout(1, score_layout, k_width=4)
    q_fragment = tlx.require_layout(q, lhs_layout, pin=False)
    do_fragment = tlx.require_layout(do, lhs_layout, pin=False)
    # A single softmax key has a zero score derivative.
    if H == 2 and MAX_KV == 1024:
        dq = tlx.zeros((BQ, 128), tl.float32, layout=score_layout)
    else:
        dq = tl.full((BQ, 128), 0, tl.float32)
    # One K buffer overlaps refill with register consumers.
    if MAX_KV > 0:
        kb = tlx.local_alloc((KL, 128), tl.bfloat16, 1)
        first_columns = ks + tl.arange(0, KL)
        first_offsets = (first_columns[:, None] * H + head) * 128 + d[None, :]
        first_token = tlx.async_load(K + first_offsets, tlx.local_view(kb, 0), mask=(first_columns < ke)[:, None],
                                     other=0)
        tlx.async_load_commit_group([first_token])
        for tile in range(triton.cdiv(MAX_KV, KL)):
            columns = ks + tile * KL + tl.arange(0, KL)
            km = columns < ke
            # Wait before reading the active buffer.
            wait = tlx.async_load_wait_group(0)
            k = tlx.local_load(tlx.local_view(kb, 0), token=wait)
            # The final iteration starts no copy.
            if tile + 1 < triton.cdiv(MAX_KV, KL):
                next_columns = ks + (tile + 1) * KL + tl.arange(0, KL)
                next_offsets = (next_columns[:, None] * H + head) * 128 + d[None, :]
                next_token = tlx.async_load(K + next_offsets, tlx.local_view(kb, 0), mask=(next_columns < ke)[:, None],
                                            other=0)
                tlx.async_load_commit_group([next_token])
            k_fragment = tlx.require_layout(tl.trans(k), rhs_layout, pin=False)
            zero_score = tlx.zeros((BQ, KL), tl.float32, layout=score_layout)
            scores = tlx.release_layout(tl.dot(q_fragment, k_fragment, zero_score, allow_tf32=False))
            p = tl.where(ke - ks == 1, 1.0, tl.exp2(scores * (alpha * 1.44269504) - m[:, None]))
            dp = tlx.release_layout(tl.dot(do_fragment, k_fragment, zero_score, allow_tf32=False))
            ds = tl.where(qm[:, None] & km[None, :] & (ke - ks > 1), p * (dp - delta[:, None]), 0).to(tl.bfloat16)
            if H == 2 and MAX_KV == 1024:
                ds_fragment = tlx.require_layout(ds, lhs_layout, pin=False)
                k_dq_fragment = tlx.require_layout(k, rhs_layout, pin=False)
                dq = tl.dot(ds_fragment, k_dq_fragment, dq, allow_tf32=False)
            else:
                dq = tl.dot(ds, k, dq, allow_tf32=False)
    if H == 2 and MAX_KV == 1024:
        dq = tlx.release_layout(dq)
    tl.store(DQ + qo, dq * alpha, qm[:, None])


@triton.jit
def _compact_q_owner(Q, K, V, DO, M, OUT, DQ, qs, qe, ks, ke, head, task, alpha, H: tl.constexpr, MAX_KV: tl.constexpr,
                     SHARED: tl.constexpr, BQ: tl.constexpr, KL: tl.constexpr):
    rows = qs + task * BQ + tl.arange(0, BQ)
    d = tl.arange(0, 128)
    qm = rows < qe
    qo = (rows[:, None] * H + head) * 128 + d[None, :]
    q = tl.load(Q + qo, qm[:, None], other=0)
    do = tl.load(DO + qo, qm[:, None], other=0)
    out = tl.load(OUT + qo, qm[:, None], other=0)
    m = tl.load(M + rows * H + head, qm, other=0)
    delta = tl.sum(out.to(tl.float32) * do.to(tl.float32), 1)
    score_layout: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                     warps_per_cta=[1, 4])
    lhs_layout: tl.constexpr = tlx.dot_operand_layout(0, score_layout, k_width=4)
    rhs_layout: tl.constexpr = tlx.dot_operand_layout(1, score_layout, k_width=4)
    q_fragment = tlx.require_layout(q, lhs_layout, pin=False)
    do_fragment = tlx.require_layout(do, lhs_layout, pin=False)
    # A single softmax key has a zero score derivative.
    dq = tl.full((BQ, 128), 0, tl.float32)
    for tile in range(triton.cdiv(MAX_KV, KL)):
        columns = ks + tile * KL + tl.arange(0, KL)
        km = columns < ke
        ko = (columns[:, None] * H + head) * 128 + d[None, :]
        k = tl.load(K + ko, km[:, None], other=0)
        if not SHARED: v = tl.load(V + ko, km[:, None], other=0)
        k_fragment = tlx.require_layout(tl.trans(k), rhs_layout, pin=False)
        zero_score = tlx.zeros((BQ, KL), tl.float32, layout=score_layout)
        scores = tlx.release_layout(tl.dot(q_fragment, k_fragment, zero_score, allow_tf32=False))
        p = tl.where(ke - ks == 1, 1.0, tl.exp2(scores * (alpha * 1.44269504) - m[:, None]))
        if SHARED:
            v_fragment = k_fragment
        else:
            v_fragment = tlx.require_layout(tl.trans(v), rhs_layout, pin=False)
        dp = tlx.release_layout(tl.dot(do_fragment, v_fragment, zero_score, allow_tf32=False))
        ds = tl.where(qm[:, None] & km[None, :] & (ke - ks > 1), p * (dp - delta[:, None]), 0).to(tl.bfloat16)
        dq = tl.dot(ds, k, dq, allow_tf32=False)
    tl.store(DQ + qo, dq * alpha, qm[:, None])


@triton.jit
def _compact_kv_owner(Q, K, V, DO, M, OUT, DK, DV, qs, qe, ks, ke, head, task, alpha, H: tl.constexpr,
                      SHARED: tl.constexpr, BK: tl.constexpr, QL: tl.constexpr, WIDE: tl.constexpr = False):
    columns = ks + task * BK + tl.arange(0, BK)
    d = tl.arange(0, 128)
    km = columns < ke
    ko = (columns[:, None] * H + head) * 128 + d[None, :]
    k = tl.load(K + ko, km[:, None], other=0)
    if SHARED: v = k
    else: v = tl.load(V + ko, km[:, None], other=0)
    # Distribute the two KV row groups across two warps.
    score_layout: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                     warps_per_cta=([2, 4] if WIDE else [2, 2]))
    lhs_layout: tl.constexpr = tlx.dot_operand_layout(0, score_layout, k_width=4)
    rhs_layout: tl.constexpr = tlx.dot_operand_layout(1, score_layout, k_width=4)
    k_fragment = tlx.require_layout(k, lhs_layout, pin=False)
    v_fragment = tlx.require_layout(v, lhs_layout, pin=False)
    # Preserve dV while clearing the single-key score derivative.
    if WIDE:
        dk = tlx.zeros((BK, 128), tl.float32, layout=score_layout)
    else:
        dk = tl.full((BK, 128), 0, tl.float32)
    if WIDE:
        dv = tlx.zeros((BK, 128), tl.float32, layout=score_layout)
    else:
        dv = tl.full((BK, 128), 0, tl.float32)
    for tile in range(triton.cdiv(256, QL)):
        rows = qs + tile * QL + tl.arange(0, QL)
        qm = rows < qe
        qo = (rows[:, None] * H + head) * 128 + d[None, :]
        q = tl.load(Q + qo, qm[:, None], other=0)
        do = tl.load(DO + qo, qm[:, None], other=0)
        out = tl.load(OUT + qo, qm[:, None], other=0)
        m = tl.load(M + rows * H + head, qm, other=0)
        delta = tl.sum(out.to(tl.float32) * do.to(tl.float32), 1)
        valid = km[:, None] & qm[None, :]
        q_fragment = tlx.require_layout(tl.trans(q), rhs_layout, pin=False)
        do_fragment = tlx.require_layout(tl.trans(do), rhs_layout, pin=False)
        zero_score = tlx.zeros((BK, QL), tl.float32, layout=score_layout)
        scores = tlx.release_layout(tl.dot(k_fragment, q_fragment, zero_score, allow_tf32=False))
        p = tl.where(valid, tl.where(ke - ks == 1, 1.0, tl.exp2(scores * (alpha * 1.44269504) - m[None, :])), 0)
        dp = tlx.release_layout(tl.dot(v_fragment, do_fragment, zero_score, allow_tf32=False))
        ds = tl.where(valid & (ke - ks > 1), p * (dp - delta[None, :]), 0).to(tl.bfloat16)
        if WIDE:
            p_fragment = tlx.require_layout(p.to(tl.bfloat16), lhs_layout, pin=False)
            do_dv_fragment = tlx.require_layout(do, rhs_layout, pin=False)
            dv = tl.dot(p_fragment, do_dv_fragment, dv, allow_tf32=False)
        else:
            dv = tl.dot(p.to(tl.bfloat16), do, dv, allow_tf32=False)
        if WIDE:
            ds_fragment = tlx.require_layout(ds, lhs_layout, pin=False)
            q_dk_fragment = tlx.require_layout(q, rhs_layout, pin=False)
            dk = tl.dot(ds_fragment, q_dk_fragment, dk, allow_tf32=False)
        else:
            dk = tl.dot(ds, q, dk, allow_tf32=False)
    if WIDE:
        dk = tlx.release_layout(dk)
        dv = tlx.release_layout(dv)
    dk = dk * alpha
    if SHARED: dk += dv
    tl.store(DK + ko, dk, km[:, None])
    if not SHARED: tl.store(DV + ko, dv, km[:, None])


@triton.jit
def _compact_bwd_impl(Q, K, V, DO, M, OUT, SOQ, SOK, DQ, DK, DV, alpha, H: tl.constexpr, MAX_KV: tl.constexpr,
                      SHARED: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr, QL: tl.constexpr, KL: tl.constexpr):
    bh = tl.program_id(0)
    task = tl.program_id(1)
    batch = bh // H
    head = bh % H
    qs = tl.load(SOQ + batch).to(tl.int64)
    qe = tl.load(SOQ + batch + 1).to(tl.int64)
    ks = tl.load(SOK + batch).to(tl.int64)
    ke = tl.load(SOK + batch + 1).to(tl.int64)
    if task < triton.cdiv(256, BQ):
        if SHARED and MAX_KV > 512:
            _compact_q_owner_staged(Q, K, DO, M, OUT, DQ, qs, qe, ks, ke, head, task, alpha, H, MAX_KV, BQ, KL)
        else:
            _compact_q_owner(Q, K, V, DO, M, OUT, DQ, qs, qe, ks, ke, head, task, alpha, H, MAX_KV, SHARED, BQ, KL)
    else:
        _compact_kv_owner(Q, K, V, DO, M, OUT, DK, DV, qs, qe, ks, ke, head, task - triton.cdiv(256, BQ), alpha, H,
                          SHARED, BK, QL, WIDE=SHARED and H == 2 and MAX_KV == 1024)


@triton.jit
def _compact_bwd_shared(Q, K, DO, M, OUT, SOQ, SOK, DQ, DK, alpha, H: tl.constexpr, MAX_KV: tl.constexpr):
    _compact_bwd_impl(Q, K, K, DO, M, OUT, SOQ, SOK, DQ, DK, DK, alpha, H, MAX_KV, True, 16, 32, 64, 128)


@triton.jit
def _compact_bwd_separate(Q, K, V, DO, M, OUT, SOQ, SOK, DQ, DK, DV, alpha, H: tl.constexpr, MAX_KV: tl.constexpr):
    _compact_bwd_impl(Q, K, V, DO, M, OUT, SOQ, SOK, DQ, DK, DV, alpha, H, MAX_KV, False, 16, 32, 64, 128)


class _CompactSharedLauncher(_GuardedLauncher):
    """Use only fresh empty_like outputs from compact_softmax_backward."""

    def _argument_keys(self, complete):
        q, k, do, m, out, soq, sok, dq, dk, alpha, heads, cap = complete
        if (type(q) is not torch.Tensor or type(k) is not torch.Tensor or type(dq) is not torch.Tensor
                or type(dk) is not torch.Tensor):
            return super()._argument_keys(complete)
        q_key, k_key = _argument_key(q), _argument_key(k)
        if q_key[1] != torch.bfloat16 or k_key[1] != torch.bfloat16:
            return super()._argument_keys(complete)
        # Fresh empty_like outputs retain their actual pointer alignment checks.
        return (q_key, k_key, _argument_key(do), _argument_key(m), _argument_key(out), _argument_key(soq),
                _argument_key(sok), (torch.Tensor, torch.bfloat16, q_key[2], dq.data_ptr() % 16 == 0, q.numel() * 2
                                     <= 2**31 - 1),
                (torch.Tensor, torch.bfloat16, k_key[2], dk.data_ptr() % 16 == 0, k.numel() * 2
                 <= 2**31 - 1), _scalar_key(alpha), _scalar_key(heads), _scalar_key(cap))


class _CompactSeparateLauncher(_GuardedLauncher):
    """Use only fresh empty_like outputs from compact_softmax_backward."""

    def _argument_keys(self, complete):
        q, k, v, do, m, out, soq, sok, dq, dk, dv, alpha, heads, cap = complete
        if (type(q) is not torch.Tensor or type(k) is not torch.Tensor or type(v) is not torch.Tensor
                or type(dq) is not torch.Tensor or type(dk) is not torch.Tensor or type(dv) is not torch.Tensor):
            return super()._argument_keys(complete)
        q_key, k_key, v_key = _argument_key(q), _argument_key(k), _argument_key(v)
        if q_key[1] != torch.bfloat16 or k_key[1] != torch.bfloat16 or v_key[1] != torch.bfloat16:
            return super()._argument_keys(complete)
        return (q_key, k_key, v_key, _argument_key(do), _argument_key(m), _argument_key(out), _argument_key(soq),
                _argument_key(sok), (torch.Tensor, torch.bfloat16, q_key[2], dq.data_ptr() % 16 == 0, q.numel() * 2
                                     <= 2**31 - 1),
                (torch.Tensor, torch.bfloat16, k_key[2], dk.data_ptr() % 16 == 0, k.numel() * 2
                 <= 2**31 - 1), (torch.Tensor, torch.bfloat16, v_key[2], dv.data_ptr() % 16 == 0, v.numel() * 2
                                 <= 2**31 - 1), _scalar_key(alpha), _scalar_key(heads), _scalar_key(cap))


# Two warp groups accumulate separate KV halves in FP32.
@triton.jit
def _compact_q_owner_split(Q, K, DO, M, OUT, DQ, qs, qe, ks, ke, task, alpha):
    rows = qs + task * 16 + tl.arange(0, 16)
    dims = tl.arange(0, 128)
    qm = rows < qe
    qo = rows[:, None] * 128 + dims[None, :]
    q = tl.load(Q + qo, qm[:, None], other=0)
    do = tl.load(DO + qo, qm[:, None], other=0)
    out = tl.load(OUT + qo, qm[:, None], other=0)
    m = tl.load(M + rows, qm, other=0)
    delta = tl.sum(out.to(tl.float32) * do.to(tl.float32), 1)
    mma: tl.constexpr = tlx.amd_mfma_layout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                            warps_per_cta=[2, 1, 4])
    lhs: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=4)
    rhs: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=4)
    q_fragment = tlx.require_layout(tl.broadcast_to(q[None, :, :], (2, 16, 128)), lhs, pin=False)
    do_fragment = tlx.require_layout(tl.broadcast_to(do[None, :, :], (2, 16, 128)), lhs, pin=False)
    dq = tlx.zeros((2, 16, 128), tl.float32, layout=mma)
    groups = tl.arange(0, 2)
    for tile in tl.range(0, 8, loop_unroll_factor=2):
        columns = ks + groups[:, None] * 1024 + tile * 128 + tl.arange(0, 128)[None, :]
        km = columns < ke
        ko = columns[:, :, None] * 128 + dims[None, None, :]
        k = tl.load(K + ko, km[:, :, None], other=0)
        kt_fragment = tlx.require_layout(k.permute(0, 2, 1), rhs, pin=False)
        zero_score = tlx.zeros((2, 16, 128), tl.float32, layout=mma)
        scores = tlx.release_layout(tl.dot(q_fragment, kt_fragment, zero_score, allow_tf32=False))
        p = tl.where(ke - ks == 1, 1.0, tl.exp2(scores * (alpha * 1.44269504) - m[None, :, None]))
        dp = tlx.release_layout(tl.dot(do_fragment, kt_fragment, zero_score, allow_tf32=False))
        ds = tl.where(qm[None, :, None] & km[:, None, :] & (ke - ks > 1), p * (dp - delta[None, :, None]),
                      0).to(tl.bfloat16)
        ds_fragment = tlx.require_layout(ds, lhs, pin=False)
        k_fragment = tlx.require_layout(k, rhs, pin=False)
        dq = tl.dot(ds_fragment, k_fragment, dq, allow_tf32=False)
    result = tl.sum(tlx.release_layout(dq), 0)
    tl.store(DQ + qo, result * alpha, qm[:, None])


@triton.jit
def _compact_bwd_shared_split(Q, K, DO, M, OUT, SOQ, SOK, DQ, DK, alpha, H: tl.constexpr, MAX_KV: tl.constexpr):
    batch = tl.program_id(0)
    task = tl.program_id(1)
    qs = tl.load(SOQ + batch).to(tl.int64)
    qe = tl.load(SOQ + batch + 1).to(tl.int64)
    ks = tl.load(SOK + batch).to(tl.int64)
    ke = tl.load(SOK + batch + 1).to(tl.int64)
    if task < 16:
        _compact_q_owner_split(Q, K, DO, M, OUT, DQ, qs, qe, ks, ke, task, alpha)
    else:
        _compact_kv_owner(Q, K, K, DO, M, OUT, DK, DK, qs, qe, ks, ke, 0, task - 16, alpha, 1, True, 32, 64, WIDE=True)


_COMPACT_SPLIT_OPTIONS = dict(_SMALL_OPTIONS, num_warps=8)
_compact_split_launcher = _CompactSharedLauncher(_compact_bwd_shared_split, bound_options=_COMPACT_SPLIT_OPTIONS)

_COMPACT_H2_1024_OPTIONS = dict(_SMALL_OPTIONS, num_warps=8)
_compact_h2_1024_launcher = _CompactSharedLauncher(_compact_bwd_shared, bound_options=_COMPACT_H2_1024_OPTIONS)
_compact_shared_launcher = _CompactSharedLauncher(_compact_bwd_shared, bound_options=_SMALL_OPTIONS)
_compact_separate_launcher = _CompactSeparateLauncher(_compact_bwd_separate, bound_options=_SMALL_OPTIONS)


def compact_softmax_backward(q, k, v, dout, m, out, offsets_q, offsets_kv, max_kv, alpha, shared_kv):
    dout = dout.contiguous()
    heads = q.shape[1]
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = None if shared_kv else torch.empty_like(v)
    grid = ((offsets_q.numel() - 1) * heads, 16 + triton.cdiv(max_kv, 32), 1)
    if shared_kv:
        if heads == 1 and max_kv == 2048 and offsets_q.numel() == 5:
            _compact_split_launcher.launch(grid,
                                           (q, k, dout, m, out, offsets_q, offsets_kv, dq, dk, alpha, heads, max_kv),
                                           _COMPACT_SPLIT_OPTIONS)
        elif heads == 2 and max_kv == 1024:
            _compact_h2_1024_launcher.launch(grid,
                                             (q, k, dout, m, out, offsets_q, offsets_kv, dq, dk, alpha, heads, max_kv),
                                             _COMPACT_H2_1024_OPTIONS)
        else:
            _compact_shared_launcher.launch(grid,
                                            (q, k, dout, m, out, offsets_q, offsets_kv, dq, dk, alpha, heads, max_kv),
                                            _SMALL_OPTIONS)
    else:
        _compact_separate_launcher.launch(
            grid, (q, k, v, dout, m, out, offsets_q, offsets_kv, dq, dk, dv, alpha, heads, max_kv), _SMALL_OPTIONS)
    return dq, dk, dv
