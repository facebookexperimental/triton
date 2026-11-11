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
"""gfx950 HSTU cross attention -- the ``tlx.ops.hstu_cross_attn`` backend.

Promoted from ``tutorials/hstu_cross_attn`` together with the optimizations
made to the kernel after that snapshot:

  - longest-KV-first launch order and the XCD regroup gate;
  - KV address arithmetic hoisted out of the KV loop;
  - 16-byte direct-to-LDS K/V loads for the shared-KV HEAD_DIM=128 path.

The softmax rescales lazily (`_forward_softmax_lazy`) only for BF16 with
distinct K/V outside causal attention, on the double-buffered path. On gfx950
a packed-FP32 subtract can read a wrong operand while another wave on the same
SIMD runs ``v_mfma_f32_16x16x32_f16``, producing NaN/Inf rows at two waves per
SIMD; the lazy BF16 kernels issue neither, folding the subtract into a packed
FMA.

The custom ops live in the ``tlx`` namespace, the environment knobs are
prefixed ``TLX_HSTU_CROSS_ATTN_``. ``space="curated"`` (the default) autotunes
the forward over a short config list for head dims that have one
(``_FWD_CURATED_BY_HEAD_DIM``) and over the whole 144-config grid otherwise;
``space="full"`` always tunes the grid, and ``space="smoke"`` runs one config
per kernel for correctness tests.

Design notes on the forward follow.

The kernel derives from the Triton HSTU cross attention
(``tutorials/hstu_cross_attn/triton_bw_cross_attention.py``) as a forward-only,
TMA-free copy, with the KV load path staged through LDS as in the TLX HSTU
self attention:

  - **K and V are staged through LDS.** `lds_k` / `lds_v` are
    `tlx.local_alloc`'d once outside the KV loop; each block runs
    `tlx.async_load` -> `async_load_commit_group` -> `async_load_wait_group(0)`
    -> `tlx.local_load`. Both copies go into one async group, so their global
    latencies overlap instead of being paid back to back.
  - **K is transposed in LDS.** It is fetched row-major `(BLOCK_N, HEAD_DIM)`
    and transposed with `tlx.local_trans`, a metadata-only view, replacing the
    register-level `tl.trans(k)`. The source already read K contiguously, so
    what LDS buys here is the register round trip, not the access pattern.
  - **XCD remapping.** The grid becomes 1-D
    (`cdiv(max_q_len, BLOCK_M) * Z * H`) and, when there are at least 8
    (sequence, head) pairs per XCD, `remap_xcd` regroups tiles so they
    are contiguous within each of the 8 XCDs instead of round-robin across
    them. With fewer pairs the round-robin order is kept, since grouping
    whole sequences per XCD then leaves XCDs idle. Adds a `MAX_Q_LEN`
    runtime arg for the in-kernel tile count, and
    moves the program-id decode out of `_compute_offsets`, which now takes
    `off_z` / `off_h` as arguments.

The main-loop and last-block bodies, which the source repeated inline, are
merged into one `_tlx_gfx950_cross_attn_fwd_one_block` taking a `MASKED`
constexpr. They differed only in the bounds mask, and the staging sequence
would otherwise have been written twice. The masked and unmasked forms are
numerically identical to the source's two bodies: for softmax the source set
invalid entries to `-inf` before scaling by `alpha` where
`forward_softmax_activation_scaled_alpha` scales first and masks second, and
for silu it zeroed `qk` before scaling where the helper zeroes `alpha` -- both
land on the same value.

`TRANS` is dropped, along with `forward_valid_mask_trans`,
`forward_epilogue_trans` and `forward_softmax_activation_trans_scaled_alpha`.
All 144 configs in the AMD list set it False, so `TRANS=True` was unreachable
on gfx950, and staging it would have meant a second copy of the load sequence
for a path nothing selects. `SHARED_KV` and `IS_DYNAMIC` are kept.

Under `SHARED_KV` the tile is staged once and read back through both
orientations, so the KV loop issues one `async_load` per block instead of two.
Layout inference cannot give a single `local_alloc` both a `tlx.local_trans`'d
and an untransposed consumer -- it widens the two to `UNKNOWN` and
`TlxPropagateLayout` then fails the module -- so the layout is pinned
explicitly, which skips inference. A padded layout at `(128, 16)` is what makes
the pair cheap; the choice is not smooth, and `(128, 4)` is 100% slower while
intervals below 128 fail to lower at all.

Halving the fetches is not where most of the win is. What the single fetch buys
is the room to double-buffer: the KV loop keeps two staged tiles and issues the
fetch for block i+1 before block i's dots, so the global latency overlaps the
MFMA chain instead of serialising behind it. Single-buffering cannot do that
here, because under `SHARED_KV` the PV dot reads the same buffer as the QK dot,
so the tile stays live to the end of the block and `async_load_wait_group(0)`
has to drain before the next fetch can start. The two buffers put LDS back to
what the distinct-K/V path already used, so this is not a new occupancy cost.

Double-buffering needs a workgroup barrier per iteration, and it is
load-bearing rather than defensive: iteration i+1 fetches into the buffer
iteration i has just finished reading, and waves within the workgroup are free
to skew, so without the rendezvous a fast wave can overwrite a tile a slow wave
is still consuming. The tail is drained with an explicit
`async_load_wait_group(0)` before buffer 0 is reused for the masked last block,
so the unconsumed prefetch cannot land on top of it.

The pin is gated on `SHARED_KV` because it is a cost on its own: forcing it
with distinct K and V, where each buffer would otherwise get its own inferred
layout, measures ~3% slower. Reusing the tile is worth more than that, so the
two together are a net win -- see the test plan.

Output is no longer bit-identical to the upstream kernel, and cannot be:
`local_trans` changes the MFMA operand layout, which changes the accumulation
order.
"""

import functools
import os
from typing import List, Optional, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl

# @manual=//triton:triton
import triton.language.extra.tlx as tlx
from ._backward_pipeline import _tlx_gfx950_cross_attn_v3_ttgir_bwd
from ._backward_small import coarse_softmax_backward, compact_softmax_backward, small_softmax_backward
from ._backward_retained import retained_softmax_backward
from ._forward_stream_k import stream_k_fwd, stream_k_supported
from ..flash_attn.gfx950 import LAZY_RESCALE_THRESHOLD
from ..hstu_attn._stubs import triton_autotune
from ..hstu_attn._util import (
    backward_softmax_activation_scaled_alpha,
    fast_silu,
    forward_softmax_activation_scaled_alpha,
)
from ._util import (
    _attn_bwd_preprocess,
    backward_common_preprocess,
    backward_d_silu_activation,
    backward_d_softmax_activation,
    backward_silu_activation,
    backward_valid_mask,
    forward_custom_vars,
    forward_epilogue,
    forward_softmax_common_preprocess,
    forward_valid_mask,
    target_common_preprocess,
    uih_common_preprocess,
)


def switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        # Tell Dynamo this data-dependent value is in the range (0, 10**9)
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10**9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def autotune_max_seq_len(runtime_max_seq_len: int) -> int:
    # Largest power of 2 <= runtime_max_seq_len (0 stays 0).
    out = next_power_of_2(runtime_max_seq_len)
    return out // 2 if out > runtime_max_seq_len else out


class _PlainFuncWrapper:
    """Stand-in for a custom op under AOTI_LOWER, so export traces through it."""

    def __init__(self, func):
        self._func = func
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def register_fake(self, fn):
        return fn

    def register_kernel(self, *args, **kwargs):
        return lambda fn: fn


def maybe_register_custom_op(op_name, mutates_args):

    def decorator(func):
        if os.environ.get("AOTI_LOWER"):
            return _PlainFuncWrapper(func)
        return torch.library.custom_op(op_name, func, mutates_args=mutates_args)

    return decorator


# "curated" autotunes over the kernel's curated config list for the call's
# regime; "full" over the whole grid; "smoke" runs one config per kernel, for
# correctness tests. Set by `hstu_cross_attn` before each launch.
_SPACE: str = "curated"


class _SpaceAutotuner:
    """Launches the curated-, full- or smoke-space autotuner of one JIT kernel."""

    def __init__(self, full, smoke, curated=None):
        self.full = full
        self.smoke = smoke
        self.curated = curated if curated is not None else full

    def select(self):
        return {"curated": self.curated, "full": self.full, "smoke": self.smoke}[_SPACE]

    def __getitem__(self, grid):
        return self.select()[grid]


def _space_autotune(configs, smoke_configs, key, curated_prune=None):
    """`curated_prune(configs, named_args, **kwargs)` narrows `configs` per call for the curated space."""
    assert len(smoke_configs) == 1, smoke_configs

    def decorator(fn):
        curated = None
        if curated_prune is not None:
            curated = triton_autotune(configs=configs, key=key, prune_configs_by={"early_config_prune":
                                                                                  curated_prune})(fn)
        return _SpaceAutotuner(
            triton_autotune(configs=configs, key=key)(fn),
            triton_autotune(configs=smoke_configs, key=key)(fn),
            curated,
        )

    return decorator


# MI300X/MI350X expose 8 XCDs; round-robin pid assignment across them hurts L2
# reuse, so tiles are re-grouped to be contiguous within an XCD.
NUM_XCDS: tl.constexpr = tl.constexpr(8)
# Grouping hands each XCD whole (sequence, head) pairs, so with few pairs one
# long sequence sets the time for its XCD while the others idle. Measured on
# MI350X: at Z*H <= 32 the regrouped grid is up to 2.3x slower than the
# round-robin one, at Z*H >= 64 it is up to 1.45x faster.
MIN_PAIRS_PER_XCD_FOR_REMAP: tl.constexpr = tl.constexpr(8)

# `autotune_max_seq_len` is the wrong quantizer for the q-length autotune key:
# its runtime path is `prev_power_of_2`, which rounds *down* and so merges
# max_q_len=160 into the same entry as 128 -- the config tuned for 128 then
# tiles 160 badly. Its static path rounds up, but into KV-sized buckets
# (1024+), which collapses every target count into one entry.
#
# 32 is the smallest BLOCK_M in `get_fwd_triton_spec_configs`, and rounding up
# to a multiple of it leaves `cdiv(max_q_len, BLOCK_M)` unchanged for every
# BLOCK_M in the sweep -- so the bucket never merges two lengths that tile
# differently.
AUTOTUNE_Q_LEN_GRANULARITY: int = 32

SPLIT_SOFTMAX_DKDV_BLOCK_M: int = 64
SPLIT_SOFTMAX_DKDV_BLOCK_N: int = 128
SPLIT_SOFTMAX_DKDV_NUM_WARPS: int = 4
SPLIT_SOFTMAX_DKDV_MATRIX_INSTR_NONKDIM: int = 16
SPLIT_SOFTMAX_DKDV_WAVES_PER_EU: int = 1
SPLIT_SOFTMAX_DKDV_NUM_STAGES: int = 1
SPLIT_SOFTMAX_DQ_BLOCK_M: int = 64
SPLIT_SOFTMAX_DQ_BLOCK_N: int = 128
SPLIT_SOFTMAX_DQ_NUM_WARPS: int = 4
SPLIT_SOFTMAX_DQ_MATRIX_INSTR_NONKDIM: int = 16
SPLIT_SOFTMAX_DQ_WAVES_PER_EU: int = 1
SPLIT_SOFTMAX_DQ_NUM_STAGES: int = 1

# Forwards that `stream_k_supported` accepts (non-causal, untruncated, separate K/V, head dim 128, at most 64
# sequences, all or no softmax heads) take the persistent stream-K kernel in `_forward_stream_k`; "0" keeps them
# on the tiled kernel. For benchmarking and A/B tests.
STREAM_K_ENV: str = "TLX_HSTU_CROSS_ATTN_STREAM_K"

# Below the XCD regroup threshold, launch the tiles of the longest-KV
# sequences first (longest-processing-time order): with ~3 waves of tiles of
# uneven length, launching in sequence order leaves CUs idle behind the long
# sequences at the end. Each program ranks the Z KV lengths itself, so the
# kernel signature is unchanged; needs Z <= SEQ_SORT_MAX_Z. Read once at import.
SEQ_SORT_MAX_Z: tl.constexpr = tl.constexpr(64)
# Only with ~2+ waves of programs, where the end-of-launch tail matters; on
# small launches the ranking is pure overhead (~2% on the production shape).
SEQ_SORT_MIN_PROGRAMS: tl.constexpr = tl.constexpr(512)
FWD_SORT_BY_KV: tl.constexpr = tl.constexpr(os.environ.get("TLX_HSTU_CROSS_ATTN_SORT_BY_KV", "1") == "1")


@functools.lru_cache(maxsize=None)
def _num_cus(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr):
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN is not divisible by NUM_XCDS, the first `tall_xcds` XCDs get
    # pids_per_xcd tiles and the rest get one fewer.
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid)
    return pid


@triton.jit
def _seq_by_kv_rank(rank, Z, seq_offsets):
    """Index of the sequence with the `rank`-th longest KV (ties by index).

    One scalar pass over the Z lengths with a SEQ_SORT_MAX_Z-wide vector of
    ranks; a Z x Z comparison tile costs enough registers to slow the main
    loop of the register-tight configs.
    """
    idx = tl.arange(0, SEQ_SORT_MAX_Z)
    valid = idx < Z
    length = tl.load(seq_offsets + idx + 1, mask=valid, other=0) - tl.load(seq_offsets + idx, mask=valid, other=0)
    ranks = tl.zeros([SEQ_SORT_MAX_Z], dtype=tl.int32)
    for j in range(Z):
        length_j = tl.load(seq_offsets + j + 1) - tl.load(seq_offsets + j)
        ahead = (length_j > length) | ((length_j == length) & (j < idx))
        ranks += ahead.to(tl.int32)
    return tl.sum(tl.where(valid & (ranks == rank), idx, 0), axis=0)


@triton.jit
def _compute_offsets(
    off_z,
    off_h,
    G,
    seq_offsets_q,
    seq_offsets,
    max_seq_len,
    TRUNCATE_METHOD: tl.constexpr,
):
    off_h_kv = off_h // G
    seq_start_kv = tl.load(seq_offsets + off_z)
    seq_end_kv = tl.load(seq_offsets + off_z + 1)
    seq_len_kv = (seq_end_kv - seq_start_kv).to(tl.int32)
    if TRUNCATE_METHOD != "none":
        truncated_len = tl.minimum(seq_len_kv, max_seq_len.to(tl.int32))
        # KEEP_LAST: skip early tokens; KEEP_FIRST: keep start unchanged
        if TRUNCATE_METHOD == "keep_last":
            seq_start_kv = seq_start_kv + (seq_len_kv - truncated_len)
        seq_len_kv = truncated_len
    seq_start_q = tl.load(seq_offsets_q + off_z)
    seq_end_q = tl.load(seq_offsets_q + off_z + 1)
    seq_len_q = (seq_end_q - seq_start_q).to(tl.int32)

    return (
        off_h_kv,
        seq_start_kv,
        seq_len_kv,
        seq_start_q,
        seq_len_q,
    )


@triton.jit
def _tlx_gfx950_cross_attn_fwd_issue_kv(
    start_n,
    seq_len_kv,
    offs_n_0,
    K_base,
    k_offs,
    stride_kn,
    buf,
    MASKED: tl.constexpr,
):
    """Issue the async fetch of one KV tile into `buf` and commit it as a group.

    `k_offs` is the tile-relative element offset, built once by the caller, so
    per tile only the scalar base advances instead of every lane recomputing
    its address.
    """
    k_ptrs = K_base + tl.cast(start_n, tl.int64) * stride_kn + k_offs
    if MASKED:
        tok = tlx.async_load(k_ptrs, buf, mask=(offs_n_0 < seq_len_kv - start_n)[:, None])
    else:
        tok = tlx.async_load(k_ptrs, buf)
    tlx.async_load_commit_group([tok])


@triton.jit
def _rescale_acc(acc, corr_2d):
    return acc * corr_2d


@triton.jit
def _forward_softmax_lazy(qk, scaled_alpha, valid_mask, m_i, acc, l_i, RESCALE_THRESHOLD: tl.constexpr):
    """Online softmax whose frame `m_i` only advances when the row max grows by more than `RESCALE_THRESHOLD`.

    Probabilities stay below 2**RESCALE_THRESHOLD, and a wave skips the acc rescale when none of its rows advance.
    `scaled_alpha` must be non-negative: the row max is taken before scaling.
    """
    m_ij = tl.max(tl.where(valid_mask, qk, -float("inf")), 1) * scaled_alpha
    advance = m_ij - m_i > RESCALE_THRESHOLD
    m_new = tl.where(advance, m_ij, m_i)
    act_qk = tl.math.exp2(tl.where(valid_mask, qk * scaled_alpha - m_new[:, None], -float("inf")))
    # One for rows that keep their frame, so a wave running the rescale for some of its rows leaves the rest exact.
    corr = tl.math.exp2(m_i - m_new)
    # The predicate is broadcast to the accumulator's shape: a row predicate does not get the accumulator's lane
    # ownership from layout propagation.
    acc = tlx.warp_predicate(tl.broadcast_to(advance[:, None], acc.shape), acc, _rescale_acc,
                             args=(tl.broadcast_to(corr[:, None], acc.shape), ))
    l_i = l_i * corr + tl.sum(act_qk, 1)
    return act_qk, acc, l_i, m_new


@triton.jit
def _tlx_gfx950_cross_attn_fwd_consume_kv(
    wait_tok,
    k_buf,
    v_buf,
    start_n,
    seq_len_q,
    seq_len_kv,
    offs_m,
    offs_n_0,
    q,
    acc,
    m_i,
    l_i,
    alpha,
    uih_len_q,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_SOFTMAX: tl.constexpr,
    MASKED: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    LAZY_RESCALE: tl.constexpr = False,
    RESCALE_THRESHOLD: tl.constexpr = 0.0,
):
    """Both dots for one KV tile.

    Under `SHARED_KV` the caller passes the same buffer as `k_buf` and `v_buf`,
    so the tile is read back through both orientations; on the distinct path
    they are the two separate staging buffers.
    """
    offs_n = offs_n_0 + start_n
    kt_local = tlx.local_trans(k_buf)
    k_dot_op = tlx.local_load(kt_local, token=wait_tok)

    qk = tl.dot(q, k_dot_op)

    if MASKED:
        valid_mask = forward_valid_mask(offs_m, offs_n, uih_len_q, seq_len_q, seq_len_kv, HAS_CAUSAL)
    elif HAS_CAUSAL:
        shifted_offs_m = offs_m + seq_len_kv - uih_len_q
        valid_mask = shifted_offs_m[:, None] >= offs_n[None, :]
    else:
        valid_mask = tl.full([BLOCK_M, BLOCK_N], 1, tl.int1)

    if IS_SOFTMAX:
        if LAZY_RESCALE:
            act_qk, acc, l_i, m_i = _forward_softmax_lazy(qk, alpha, valid_mask, m_i, acc, l_i, RESCALE_THRESHOLD)
        else:
            act_qk, acc, l_i, m_i = forward_softmax_activation_scaled_alpha(qk, alpha, valid_mask, m_i, acc, l_i)
    else:
        masked_alpha = tl.where(valid_mask, alpha, 0.0)
        qk = qk * masked_alpha
        # pyrefly: ignore [bad-argument-type]
        act_qk = fast_silu(qk, MULT_BY_X=True)

    v = tlx.local_load(v_buf, token=wait_tok)
    act_qk = act_qk.to(v.dtype)
    acc = tl.dot(act_qk, v, acc)
    return acc, l_i, m_i


@triton.jit
def _tlx_gfx950_cross_attn_fwd_one_block(
    start_n,
    seq_len_q,
    seq_len_kv,
    offs_m,
    offs_n_0,
    q,
    K_base,
    V_base,
    k_offs,
    v_offs,
    stride_kn,
    stride_vn,
    lds_k,
    lds_v,
    acc,
    m_i,
    l_i,
    alpha,
    uih_len_q,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SHARED_KV: tl.constexpr,
    IS_SOFTMAX: tl.constexpr,
    MASKED: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    mask_n = offs_n_0 < seq_len_kv - start_n

    # K and V are copied straight into LDS instead of being read into registers
    # and handed to the MFMA from there. Unlike the self-attention kernel, the
    # global read this replaces was already contiguous -- the source fetches K row-major
    # and transposes in registers -- so what LDS buys here is the register-file
    # round trip and the `tl.trans`, not a strided access.
    #
    # Both copies go into one async group. V's address does not depend on the
    # QK dot, so issuing it here costs the block one global latency rather than
    # two. Only the LDS->register read of V is deferred to its use, to keep the
    # tile out of registers across the activation.
    k_ptrs = K_base + tl.cast(start_n, tl.int64) * stride_kn + k_offs
    k_local = tlx.local_view(lds_k, 0)
    v_local = tlx.local_view(lds_v, 0)
    if SHARED_KV:
        # V aliases K: fetch the tile once and read it back through two layouts.
        if MASKED:
            tok_k = tlx.async_load(k_ptrs, k_local, mask=mask_n[:, None])
        else:
            tok_k = tlx.async_load(k_ptrs, k_local)
        tlx.async_load_commit_group([tok_k])
    else:
        v_ptrs = V_base + tl.cast(start_n, tl.int64) * stride_vn + v_offs
        if MASKED:
            tok_k = tlx.async_load(k_ptrs, k_local, mask=mask_n[:, None])
            tok_v = tlx.async_load(v_ptrs, v_local, mask=mask_n[:, None])
        else:
            tok_k = tlx.async_load(k_ptrs, k_local)
            tok_v = tlx.async_load(v_ptrs, v_local)
        tlx.async_load_commit_group([tok_k, tok_v])
    # The wait token is threaded into both `local_load`s in the consumer below.
    # Without it AMD lowering keeps its own conservative producer-to-consumer
    # wait-count tracking on top of this explicit wait.
    # pyrefly: ignore [bad-argument-type]
    wait_tok = tlx.async_load_wait_group(0)
    if SHARED_KV:
        v_buf = k_local
    else:
        v_buf = v_local
    return _tlx_gfx950_cross_attn_fwd_consume_kv(
        wait_tok,
        k_local,
        v_buf,
        start_n,
        seq_len_q,
        seq_len_kv,
        offs_m,
        offs_n_0,
        q,
        acc,
        m_i,
        l_i,
        alpha,
        uih_len_q,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        IS_SOFTMAX=IS_SOFTMAX,
        MASKED=MASKED,
        HAS_CAUSAL=HAS_CAUSAL,
    )


@triton.jit
def _tlx_gfx950_cross_attn_fwd_issue_distinct_kv(
    start_n,
    seq_len_kv,
    offs_n_0,
    K_base,
    V_base,
    k_offs,
    v_offs,
    stride_kn,
    stride_vn,
    k_buf,
    v_buf,
):
    # See _issue_kv: the tile-relative offsets are loop invariant.
    k_ptrs = K_base + tl.cast(start_n, tl.int64) * stride_kn + k_offs
    v_ptrs = V_base + tl.cast(start_n, tl.int64) * stride_vn + v_offs
    mask_n = (offs_n_0 < seq_len_kv - start_n)[:, None]
    tok_k = tlx.async_load(k_ptrs, k_buf, mask=mask_n)
    tok_v = tlx.async_load(v_ptrs, v_buf, mask=mask_n)
    tlx.async_load_commit_group([tok_k, tok_v])


@triton.jit
def _tlx_gfx950_cross_attn_fwd_compute_distinct_pipeline(
    acc,
    alpha,
    q,
    K,
    V,
    offs_m,
    offs_n_0,
    off_h,
    off_h_kv,
    seq_start_kv,
    seq_len_kv,
    seq_start_q,
    seq_len_q,
    attn_scale,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_mm,
    m_i,
    l_i,
    M,
    start_m,
    uih_len_q,
    num_softmax_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    SHARED_KV: tl.constexpr,
    IS_SOFTMAX: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    LAZY_RESCALE: tl.constexpr,
    RESCALE_THRESHOLD: tl.constexpr,
):
    # Distinct-KV double-buffered pipeline covering both softmax and silu.
    # Called only when not SHARED_KV and NUM_STAGES >= 2 (see dispatcher).
    tl.static_assert(not SHARED_KV)
    if HAS_CAUSAL:
        high = start_m + BLOCK_M + seq_len_kv - uih_len_q
        high = tl.cdiv(min(high, seq_len_kv), BLOCK_N) * BLOCK_N
        last_block_start_n = tl.maximum(0, (tl.cdiv(high, BLOCK_N) - 1) * BLOCK_N)
    else:
        last_block_start_n = tl.maximum(0, (tl.cdiv(seq_len_kv, BLOCK_N) - 1) * BLOCK_N)
    if IS_SOFTMAX:
        alpha = alpha * 1.44269504
        if LAZY_RESCALE:
            # The lazy softmax takes the row max before scaling, so it needs a non-negative scale. Negating q
            # negates every score exactly.
            if alpha < 0:
                q = -q
                alpha = -alpha
    K_base = K + seq_start_kv.to(tl.int64) * stride_kn + off_h_kv * stride_kh
    V_base = V + seq_start_kv.to(tl.int64) * stride_vn + off_h_kv * stride_vh
    # Two physical slots, separately inferred per tensor. Selected by
    # NUM_STAGES >= 2 in the dispatcher; NUM_STAGES == 1 falls through to the
    # single-buffered _one_block path so stages=1/2 are not duplicates.
    lds_k = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(K), 2)
    lds_v = tlx.local_alloc((BLOCK_N, BLOCK_D_V), tlx.dtype_of(V), 2)
    k_offs = offs_n_0[:, None] * stride_kn + tl.arange(0, HEAD_DIM)[None, :]
    v_offs = offs_n_0[:, None] * stride_vn + tl.arange(0, BLOCK_D_V)[None, :]
    _tlx_gfx950_cross_attn_fwd_issue_distinct_kv(
        0,
        seq_len_kv,
        offs_n_0,
        K_base,
        V_base,
        k_offs,
        v_offs,
        stride_kn,
        stride_vn,
        tlx.local_view(lds_k, 0),
        tlx.local_view(lds_v, 0),
    )
    for start_n in tl.range(0, last_block_start_n, BLOCK_N):
        slot = (start_n // BLOCK_N) % 2
        _tlx_gfx950_cross_attn_fwd_issue_distinct_kv(
            start_n + BLOCK_N,
            seq_len_kv,
            offs_n_0,
            K_base,
            V_base,
            k_offs,
            v_offs,
            stride_kn,
            stride_vn,
            tlx.local_view(lds_k, 1 - slot),
            tlx.local_view(lds_v, 1 - slot),
        )
        wait_tok = tlx.async_load_wait_group(1)
        acc, l_i, m_i = _tlx_gfx950_cross_attn_fwd_consume_kv(
            wait_tok,
            tlx.local_view(lds_k, slot),
            tlx.local_view(lds_v, slot),
            start_n,
            seq_len_q,
            seq_len_kv,
            offs_m,
            offs_n_0,
            q,
            acc,
            m_i,
            l_i,
            alpha,
            uih_len_q,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            IS_SOFTMAX=IS_SOFTMAX,
            MASKED=False,
            HAS_CAUSAL=HAS_CAUSAL,
            LAZY_RESCALE=LAZY_RESCALE,
            RESCALE_THRESHOLD=RESCALE_THRESHOLD,
        )
        # Readers in every wave must finish before the next slot reuse.
        tl.debug_barrier()
    last_slot = (last_block_start_n // BLOCK_N) % 2
    wait_tok = tlx.async_load_wait_group(0)
    acc, l_i, m_i = _tlx_gfx950_cross_attn_fwd_consume_kv(
        wait_tok,
        tlx.local_view(lds_k, last_slot),
        tlx.local_view(lds_v, last_slot),
        last_block_start_n,
        seq_len_q,
        seq_len_kv,
        offs_m,
        offs_n_0,
        q,
        acc,
        m_i,
        l_i,
        alpha,
        uih_len_q,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        IS_SOFTMAX=IS_SOFTMAX,
        MASKED=True,
        HAS_CAUSAL=HAS_CAUSAL,
        LAZY_RESCALE=LAZY_RESCALE,
        RESCALE_THRESHOLD=RESCALE_THRESHOLD,
    )
    if IS_SOFTMAX:
        # Same epilogue as _tlx_gfx950_cross_attn_fwd_compute: normalize by l_i
        # and publish M. m_i/l_i were threaded through _consume_kv above.
        acc = forward_epilogue(
            acc,
            l_i,
            m_i,
            offs_m,
            seq_start_q,
            stride_mm,
            seq_len_q,
            M,
            off_h,
            num_softmax_heads,
        )
        return acc
    # Preserve scalar attn_scale after fp32 PV reduction and before output cast.
    scale = tl.load(attn_scale).to(tl.float32)
    return acc * scale


@triton.jit
def _tlx_gfx950_cross_attn_fwd_compute(
    acc,
    alpha,
    q,
    K,
    V,
    offs_m,
    offs_n_0,
    off_h,
    off_h_kv,
    seq_start_kv,
    seq_len_kv,
    seq_start_q,
    seq_len_q,
    attn_scale,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_mm,
    m_i,
    l_i,
    M,
    start_m,
    uih_len_q,
    num_softmax_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    SHARED_KV: tl.constexpr,
    IS_SOFTMAX: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    LAZY_RESCALE: tl.constexpr,
    RESCALE_THRESHOLD: tl.constexpr,
):
    # The runtime head branch compiles both specializations, even for zero heads.
    # Distinct-KV double buffering is selected by NUM_STAGES >= 2 so the
    # stages=1/2 autotune axis is not a duplicate; single-buffering keeps the
    # NUM_STAGES == 1 behavior for low-LDS configs.
    if not SHARED_KV:
        if NUM_STAGES >= 2:
            return _tlx_gfx950_cross_attn_fwd_compute_distinct_pipeline(
                acc,
                alpha,
                q,
                K,
                V,
                offs_m,
                offs_n_0,
                off_h,
                off_h_kv,
                seq_start_kv,
                seq_len_kv,
                seq_start_q,
                seq_len_q,
                attn_scale,
                stride_kn,
                stride_kh,
                stride_vn,
                stride_vh,
                stride_mm,
                m_i,
                l_i,
                M,
                start_m,
                uih_len_q,
                num_softmax_heads,
                HEAD_DIM,
                BLOCK_D_V,
                BLOCK_M,
                BLOCK_N,
                NUM_STAGES,
                SHARED_KV,
                IS_SOFTMAX,
                HAS_CAUSAL,
                LAZY_RESCALE,
                RESCALE_THRESHOLD,
            )
    if HAS_CAUSAL:
        high = start_m + BLOCK_M + seq_len_kv - uih_len_q
        high = tl.cdiv(min(high, seq_len_kv), BLOCK_N) * BLOCK_N
        last_block_start_n = tl.maximum(0, (tl.cdiv(high, BLOCK_N) - 1) * BLOCK_N)
    else:
        last_block_start_n = tl.maximum(0, (tl.cdiv(seq_len_kv, BLOCK_N) - 1) * BLOCK_N)
    if IS_SOFTMAX:
        alpha = alpha * 1.44269504

    # Head- and sequence-offset the bases once; the per-block pointers below are
    # then a BLOCK_N stride off these. `tlx.async_load` takes a pointer tensor,
    # so the source's inline `(seq_start_kv + offs_n)` arithmetic is hoisted
    # rather than repeated per block.
    K_base = K + seq_start_kv.to(tl.int64) * stride_kn + off_h_kv * stride_kh
    V_base = V + seq_start_kv.to(tl.int64) * stride_vn + off_h_kv * stride_vh
    # Tile-relative element offsets, loop invariant: per KV tile only the
    # scalar base advances.
    k_offs = offs_n_0[:, None] * stride_kn + tl.arange(0, HEAD_DIM)[None, :]
    v_offs = offs_n_0[:, None] * stride_vn + tl.arange(0, BLOCK_D_V)[None, :]

    # LDS staging, reserved once outside the KV loop. Double-buffered under
    # SHARED_KV so the next tile's fetch can overlap the current tile's dots;
    # on the distinct path NUM_STAGES >= 2 takes the double-buffered pipeline
    # above, while NUM_STAGES == 1 stays single-buffered here.
    if SHARED_KV:
        # `lds_v` aliases `lds_k`, which is HEAD_DIM wide, so a narrower
        # BLOCK_D_V would hand the PV dot a wrong-width operand and corrupt
        # `acc` silently. The two cannot diverge today -- the wrapper
        # substitutes `v = k` before the launch, so DimV is DimQ -- but
        # nothing downstream enforces that, so fail at compile time if it
        # ever stops being true.
        tl.static_assert(BLOCK_D_V == HEAD_DIM)
        # V aliases K, so one buffer serves both dots. That means one
        # allocation read through two orientations, which layout inference
        # cannot reconcile, so the layout is pinned explicitly. The pin is
        # gated on SHARED_KV: with distinct K and V each buffer keeps its own
        # inferred layout, which is better for them.
        # The padding interval sets the width of the global->LDS copy: without
        # direct-to-LDS scatter, padding may only fall between whole wave
        # writes, so the copy is at most interval / 64 elements per lane.
        # Padding every 128 elements caps it at 4 bytes; every 512 allows
        # 16-byte copies (4x fewer loads and address ops per tile). The rows
        # are permuted (0, 16, 32, 48, 1, ...) as in tlx's gfx950 flash
        # attention: with identity rows the same 512-element padding measured
        # 10-25% slower, while this order matches the old layout's LDS bank
        # conflict counts for both the K^T and the V read of the buffer.
        if HEAD_DIM == 128:
            if BLOCK_N == 64:
                # pyrefly: ignore [missing-attribute]
                kv_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
                    [(512, 16)],
                    [
                        [0, 1],
                        [0, 2],
                        [0, 4],
                        [0, 8],
                        [0, 16],
                        [0, 32],
                        [0, 64],
                        [16, 0],
                        [32, 0],
                        [1, 0],
                        [2, 0],
                        [4, 0],
                        [8, 0],
                    ],
                    [BLOCK_N, HEAD_DIM],
                )
            else:
                tl.static_assert(BLOCK_N == 32)
                # pyrefly: ignore [missing-attribute]
                kv_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_bases(
                    [(512, 16)],
                    [
                        [0, 1],
                        [0, 2],
                        [0, 4],
                        [0, 8],
                        [0, 16],
                        [0, 32],
                        [0, 64],
                        [16, 0],
                        [8, 0],
                        [1, 0],
                        [2, 0],
                        [4, 0],
                    ],
                    [BLOCK_N, HEAD_DIM],
                )
        else:
            kv_layout: tl.constexpr = (
                # pyrefly: ignore [missing-attribute]
                tlx.padded_shared_layout_encoding.with_identity_for([(128, 16)], [BLOCK_N, HEAD_DIM], [1, 0]))
        lds_k = tlx.local_alloc(
            (BLOCK_N, HEAD_DIM),
            tlx.dtype_of(K),
            # pyrefly: ignore [bad-argument-type]
            2,
            # pyrefly: ignore [bad-argument-type]
            layout=kv_layout,
        )
        lds_v = lds_k
    else:
        # pyrefly: ignore [bad-argument-type]
        lds_k = tlx.local_alloc((BLOCK_N, HEAD_DIM), tlx.dtype_of(K), 1)
        # pyrefly: ignore [bad-argument-type]
        lds_v = tlx.local_alloc((BLOCK_N, BLOCK_D_V), tlx.dtype_of(V), 1)

    if SHARED_KV:
        # Double-buffered: the fetch for tile i+1 is issued before tile i's
        # dots, so the global latency overlaps the MFMAs instead of serialising
        # behind them. Single-buffering cannot do this -- under SHARED_KV the
        # PV dot reads the same buffer as the QK dot, so the tile is live until
        # late in the block.
        #
        # The barrier is load-bearing, not decorative. Iteration i+1
        # issues into the buffer iteration i just finished reading, and waves
        # in the workgroup are free to skew, so without the rendezvous a fast
        # wave can overwrite a tile a slow wave is still consuming.
        if last_block_start_n > 0:
            _tlx_gfx950_cross_attn_fwd_issue_kv(
                0,
                seq_len_kv,
                offs_n_0,
                K_base,
                k_offs,
                stride_kn,
                tlx.local_view(lds_k, 0),
                # pyrefly: ignore [bad-argument-type]
                MASKED=False,
            )
        for start_n in tl.range(0, last_block_start_n, BLOCK_N):
            slot = (start_n // BLOCK_N) % 2
            # The prefetch is unmasked, so it must never name the last block:
            # that tile can extend past `seq_len_kv` and the fetch would read
            # out of bounds. On the tail iteration re-fetch the current block
            # instead -- interior block starts are wholly inside `seq_len_kv`,
            # and the copy is discarded either way.
            nxt_is_interior = start_n + BLOCK_N < last_block_start_n
            nxt_start = tl.where(nxt_is_interior, start_n + BLOCK_N, start_n)
            _tlx_gfx950_cross_attn_fwd_issue_kv(
                nxt_start,
                seq_len_kv,
                offs_n_0,
                K_base,
                k_offs,
                stride_kn,
                tlx.local_view(lds_k, 1 - slot),
                # pyrefly: ignore [bad-argument-type]
                MASKED=False,
            )
            # One group (the prefetch just issued) may stay outstanding.
            # pyrefly: ignore [bad-argument-type]
            wait_tok = tlx.async_load_wait_group(1)
            acc, l_i, m_i = _tlx_gfx950_cross_attn_fwd_consume_kv(
                wait_tok,
                tlx.local_view(lds_k, slot),
                tlx.local_view(lds_k, slot),
                start_n,
                seq_len_q,
                seq_len_kv,
                offs_m,
                offs_n_0,
                q,
                acc,
                m_i,
                l_i,
                alpha,
                uih_len_q,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                IS_SOFTMAX=IS_SOFTMAX,
                # pyrefly: ignore [bad-argument-type]
                MASKED=False,
                HAS_CAUSAL=HAS_CAUSAL,
            )
            tl.debug_barrier()

        # Drain the unconsumed tail prefetch before reusing buffer 0, so that
        # stale copy cannot land on top of the masked tile fetched below.
        # pyrefly: ignore [bad-argument-type]
        _ = tlx.async_load_wait_group(0)
        tl.debug_barrier()
        _tlx_gfx950_cross_attn_fwd_issue_kv(
            last_block_start_n,
            seq_len_kv,
            offs_n_0,
            K_base,
            k_offs,
            stride_kn,
            tlx.local_view(lds_k, 0),
            # pyrefly: ignore [bad-argument-type]
            MASKED=True,
        )
        # pyrefly: ignore [bad-argument-type]
        wait_tok = tlx.async_load_wait_group(0)
        acc, l_i, m_i = _tlx_gfx950_cross_attn_fwd_consume_kv(
            wait_tok,
            tlx.local_view(lds_k, 0),
            tlx.local_view(lds_k, 0),
            last_block_start_n,
            seq_len_q,
            seq_len_kv,
            offs_m,
            offs_n_0,
            q,
            acc,
            m_i,
            l_i,
            alpha,
            uih_len_q,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            IS_SOFTMAX=IS_SOFTMAX,
            # pyrefly: ignore [bad-argument-type]
            MASKED=True,
            HAS_CAUSAL=HAS_CAUSAL,
        )
    else:
        # Main loop - no masking needed for non-last kv blocks
        for start_n in tl.range(0, last_block_start_n, BLOCK_N):
            acc, l_i, m_i = _tlx_gfx950_cross_attn_fwd_one_block(
                start_n=start_n,
                seq_len_q=seq_len_q,
                seq_len_kv=seq_len_kv,
                offs_m=offs_m,
                offs_n_0=offs_n_0,
                q=q,
                K_base=K_base,
                V_base=V_base,
                k_offs=k_offs,
                v_offs=v_offs,
                stride_kn=stride_kn,
                stride_vn=stride_vn,
                lds_k=lds_k,
                lds_v=lds_v,
                acc=acc,
                m_i=m_i,
                l_i=l_i,
                alpha=alpha,
                uih_len_q=uih_len_q,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                SHARED_KV=SHARED_KV,
                IS_SOFTMAX=IS_SOFTMAX,
                # pyrefly: ignore [bad-argument-type]
                MASKED=False,
                HAS_CAUSAL=HAS_CAUSAL,
            )

        # Last kv block - needs masking for out-of-bounds elements;
        # q masking is not needed here because we mask when storing results.
        acc, l_i, m_i = _tlx_gfx950_cross_attn_fwd_one_block(
            start_n=last_block_start_n,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            offs_m=offs_m,
            offs_n_0=offs_n_0,
            q=q,
            K_base=K_base,
            V_base=V_base,
            k_offs=k_offs,
            v_offs=v_offs,
            stride_kn=stride_kn,
            stride_vn=stride_vn,
            lds_k=lds_k,
            lds_v=lds_v,
            acc=acc,
            m_i=m_i,
            l_i=l_i,
            alpha=alpha,
            uih_len_q=uih_len_q,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            SHARED_KV=SHARED_KV,
            IS_SOFTMAX=IS_SOFTMAX,
            # pyrefly: ignore [bad-argument-type]
            MASKED=True,
            HAS_CAUSAL=HAS_CAUSAL,
        )

    # epilogue
    if IS_SOFTMAX:
        acc = forward_epilogue(
            acc,
            l_i,
            m_i,
            offs_m,
            seq_start_q,
            stride_mm,
            seq_len_q,
            M,
            off_h,
            num_softmax_heads,
        )
    else:
        scale = tl.load(attn_scale).to(tl.float32)
        acc = acc * scale
    return acc


@triton.jit
def _tlx_gfx950_cross_attn_fwd_inner(
    alpha,
    Z,
    Q,
    K,
    V,
    Out,
    seq_offsets_q,
    seq_offsets,
    max_seq_len,
    attn_scale,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    start_m,
    off_h,
    off_h_kv,
    seq_start_kv,
    seq_len_kv,
    seq_start_q,
    seq_len_q,
    M,
    stride_mm,
    uih_len_q,
    num_softmax_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    NUM_STAGES: tl.constexpr,
    SHARED_KV: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    LAZY_RESCALE: tl.constexpr,
    RESCALE_THRESHOLD: tl.constexpr,
):
    # initialize offsets
    if start_m < seq_len_q:
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_n_0 = tl.arange(0, BLOCK_N)
        qo_offset_y_split = seq_start_q + start_m
        offs_qk_d = tl.arange(0, HEAD_DIM)
        q_ptrs = (Q + (qo_offset_y_split + tl.arange(0, BLOCK_M)).to(tl.int64)[:, None] * stride_qm +
                  (off_h * stride_qh + offs_qk_d)[None, :])
        q = tl.load(
            q_ptrs,
            mask=(offs_m < seq_len_q)[:, None],
            other=0.0,
        )
        m_i, l_i = forward_softmax_common_preprocess(off_h, num_softmax_heads, BLOCK_M)
        acc = tl.zeros([BLOCK_M, BLOCK_D_V], dtype=tl.float32)
        # when seq_len_kv is 0, we can skip computation but still needs to write acc to out
        # to avoid NaN.
        if seq_len_kv > 0:
            if off_h < num_softmax_heads:
                acc = _tlx_gfx950_cross_attn_fwd_compute(
                    acc,
                    alpha,
                    q,
                    K,
                    V,
                    offs_m,
                    offs_n_0,
                    off_h,
                    off_h_kv,
                    seq_start_kv,
                    seq_len_kv,
                    seq_start_q,
                    seq_len_q,
                    attn_scale,
                    stride_kn,
                    stride_kh,
                    stride_vn,
                    stride_vh,
                    stride_mm,
                    m_i,
                    l_i,
                    M,
                    start_m,
                    uih_len_q,
                    num_softmax_heads,
                    HEAD_DIM,
                    BLOCK_D_V,
                    BLOCK_M,
                    BLOCK_N,
                    NUM_STAGES,
                    SHARED_KV,
                    # pyrefly: ignore [bad-argument-type]
                    IS_SOFTMAX=True,
                    HAS_CAUSAL=HAS_CAUSAL,
                    LAZY_RESCALE=LAZY_RESCALE,
                    RESCALE_THRESHOLD=RESCALE_THRESHOLD,
                )
            else:
                acc = _tlx_gfx950_cross_attn_fwd_compute(
                    acc,
                    alpha,
                    q,
                    K,
                    V,
                    offs_m,
                    offs_n_0,
                    off_h,
                    off_h_kv,
                    seq_start_kv,
                    seq_len_kv,
                    seq_start_q,
                    seq_len_q,
                    attn_scale,
                    stride_kn,
                    stride_kh,
                    stride_vn,
                    stride_vh,
                    stride_mm,
                    m_i,
                    l_i,
                    M,
                    start_m,
                    uih_len_q,
                    num_softmax_heads,
                    HEAD_DIM,
                    BLOCK_D_V,
                    BLOCK_M,
                    BLOCK_N,
                    NUM_STAGES,
                    SHARED_KV,
                    # pyrefly: ignore [bad-argument-type]
                    IS_SOFTMAX=False,
                    HAS_CAUSAL=HAS_CAUSAL,
                    LAZY_RESCALE=False,
                    RESCALE_THRESHOLD=RESCALE_THRESHOLD,
                )
        off_o = Out + seq_start_q * stride_om + off_h * stride_oh
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_v_d = tl.arange(0, BLOCK_D_V)
        out_ptrs = off_o + offs_m[:, None] * stride_om + offs_v_d[None, :]
        acc = acc.to(Out.dtype.element_ty)
        tl.store(out_ptrs, acc, mask=(offs_m < seq_len_q)[:, None])


def get_fwd_triton_spec_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_M in [32, 64, 128]:
        for BLOCK_N in [32, 64]:
            for num_stages in [1, 2]:
                for num_warps in [4, 8]:
                    for matrix_instr_nonkdim in [16, 32]:
                        for waves_per_eu in [1, 2, 3]:
                            configs.append(
                                triton.Config(
                                    {
                                        "BLOCK_M": BLOCK_M,
                                        "BLOCK_N": BLOCK_N,
                                        "BLOCK_M1": BLOCK_M,
                                        "BLOCK_N1": BLOCK_N,
                                        "NUM_STAGES": num_stages,
                                        "NUM_STAGES1": num_stages,
                                        "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                        "waves_per_eu": waves_per_eu,
                                        "IS_DYNAMIC": False,
                                    },
                                    num_stages=num_stages,
                                    num_warps=num_warps,
                                ))
    return configs


# Curated forward configs for head dims 64 and 128, picked from the 144-config
# grid below. On a MI350X run not used to pick them, the best of these eight was
# within 3.6% of the grid's best on every gfx950 focus shape (_shapes.py, all
# head dim 64 or 128), and within 2.7% on shapes outside that list (GQA, short
# q, other batches). At head dims 16, 32 and 256 they were up to 18% behind and
# each new shape wanted different configs, so those head dims tune the whole
# grid until they have shapes of their own to pick a list from. Fields: BLOCK_M, BLOCK_N, NUM_STAGES, matrix_instr_nonkdim,
# waves_per_eu, num_warps. test_autotune_drift.py re-checks them.
_FWD_CURATED = frozenset({
    (64, 64, 1, 16, 3, 4),
    (64, 64, 2, 16, 1, 4),
    (64, 64, 2, 16, 2, 4),
    (64, 64, 2, 16, 3, 4),
    (128, 32, 1, 16, 3, 4),
    (128, 64, 2, 16, 2, 4),
    (128, 64, 2, 32, 1, 4),
    (128, 64, 2, 32, 2, 4),
})
_FWD_CURATED_BY_HEAD_DIM = {64: _FWD_CURATED, 128: _FWD_CURATED}


def _fwd_config_key(c: triton.Config) -> tuple:
    return (c.kwargs["BLOCK_M"], c.kwargs["BLOCK_N"], c.num_stages, c.kwargs["matrix_instr_nonkdim"],
            c.kwargs["waves_per_eu"], c.num_warps)


def _fwd_curated_configs() -> List[triton.Config]:
    return [c for c in get_fwd_triton_spec_configs() if _fwd_config_key(c) in _FWD_CURATED]


def _prune_fwd_to_curated(configs, named_args, **kwargs):
    curated = _FWD_CURATED_BY_HEAD_DIM.get(kwargs["HEAD_DIM"])
    if curated is None:  # no curated list for this head dim: tune the whole grid
        return configs
    return [c for c in configs if _fwd_config_key(c) in curated]


def _fwd_smoke_configs() -> List[triton.Config]:
    return [
        c for c in get_fwd_triton_spec_configs()
        if c.kwargs["BLOCK_M"] == 64 and c.kwargs["BLOCK_N"] == 64 and c.num_stages == 1 and c.num_warps == 4
        and c.kwargs["matrix_instr_nonkdim"] == 16 and c.kwargs["waves_per_eu"] == 2
    ]


@_space_autotune(
    configs=get_fwd_triton_spec_configs(),
    smoke_configs=_fwd_smoke_configs(),
    curated_prune=_prune_fwd_to_curated,
    key=[
        "AUTOTUNE_Z",
        "HEAD_DIM",
        # max_q_len sets BLOCK_M's tiling and the grid, so a config tuned at one
        # target count is not transferable to another. Bucketed by
        # AUTOTUNE_Q_LEN_GRANULARITY, not autotune_max_seq_len -- see the note
        # on that constant.
        "AUTOTUNE_MAX_Q_LEN",
        "AUTOTUNE_MAX_SEQ_LEN",
        "num_softmax_heads",
    ],
)
@triton.jit
def _tlx_gfx950_cross_attn_fwd(
    alpha,
    Z,
    H,
    G,
    Q,
    K,
    V,
    Out,
    seq_offsets_q,
    seq_offsets,
    max_seq_len,
    attn_scale,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    M,
    stride_mm,
    num_targets,
    MAX_Q_LEN,
    AUTOTUNE_Z,
    AUTOTUNE_MAX_Q_LEN,
    AUTOTUNE_MAX_SEQ_LEN,
    num_softmax_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    NUM_STAGES: tl.constexpr,
    NUM_STAGES1: tl.constexpr,
    SHARED_KV: tl.constexpr,
    IS_DYNAMIC: tl.constexpr,
    TRUNCATE_METHOD: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
    # Lazily rescaled softmax on the distinct-KV double-buffered path; see `_forward_softmax_lazy`.
    LAZY_RESCALE: tl.constexpr,
    RESCALE_THRESHOLD: tl.constexpr,
):
    # 1-D grid so tiles can be regrouped per XCD. The 2-D (m_tile, Z*H) launch
    # this replaces hands consecutive tiles to different XCDs, which scatters
    # workgroups that share K/V across all 8 L2s.
    tpid = tl.program_id(0)
    num_tiles = tl.cdiv(MAX_Q_LEN, BLOCK_M)
    if Z * H >= MIN_PAIRS_PER_XCD_FOR_REMAP * NUM_XCDS:
        tpid = remap_xcd(tpid, num_tiles * Z * H, NUM_XCDS)
    off_h = tpid % H
    off_nz = tpid // H
    start_m = (off_nz % num_tiles) * BLOCK_M
    off_z = off_nz // num_tiles
    if FWD_SORT_BY_KV:
        if Z * H < MIN_PAIRS_PER_XCD_FOR_REMAP * NUM_XCDS:
            if Z <= SEQ_SORT_MAX_Z:
                if num_tiles * Z * H >= SEQ_SORT_MIN_PROGRAMS:
                    off_z = _seq_by_kv_rank(off_z, Z, seq_offsets)
    (
        off_h_kv,
        seq_start_kv,
        seq_len_kv,
        seq_start_q,
        seq_len_q,
    ) = _compute_offsets(off_z, off_h, G, seq_offsets_q, seq_offsets, max_seq_len, TRUNCATE_METHOD)
    if HAS_CAUSAL:
        n_targets = target_common_preprocess(off_z, num_targets, HAS_NUM_TARGETS)
        uih_len_q = uih_common_preprocess(n_targets, seq_len_q, HAS_NUM_TARGETS)
    else:
        uih_len_q = seq_len_q
    # DO NOT merge IS_DYNAMIC into a compound boolean expression
    # (e.g. "if IS_DYNAMIC and condition:"). The Triton CC compiler
    # passes constexprs as i32 and cannot bitcast i32 to i1 for the
    # AND operator. Keeping IS_DYNAMIC as a standalone constexpr
    # branch allows dead-code elimination without the bitcast.
    if IS_DYNAMIC:
        if start_m + BLOCK_M1 >= seq_len_q:
            # Use the smaller (BLOCK_M1, BLOCK_N1) tiling for tail handling
            _tlx_gfx950_cross_attn_fwd_inner(
                alpha,
                Z,
                Q,
                K,
                V,
                Out,
                seq_offsets_q,
                seq_offsets,
                max_seq_len,
                attn_scale,
                stride_qm,
                stride_qh,
                stride_kn,
                stride_kh,
                stride_vn,
                stride_vh,
                stride_om,
                stride_oh,
                start_m,
                off_h,
                off_h_kv,
                seq_start_kv,
                seq_len_kv,
                seq_start_q,
                seq_len_q,
                M,
                stride_mm,
                uih_len_q,
                num_softmax_heads,
                HEAD_DIM,
                BLOCK_D_V,
                BLOCK_M1,
                BLOCK_N1,
                NUM_STAGES1,
                SHARED_KV=SHARED_KV,
                HAS_CAUSAL=HAS_CAUSAL,
                LAZY_RESCALE=LAZY_RESCALE,
                RESCALE_THRESHOLD=RESCALE_THRESHOLD,
            )
            return
    _tlx_gfx950_cross_attn_fwd_inner(
        alpha,
        Z,
        Q,
        K,
        V,
        Out,
        seq_offsets_q,
        seq_offsets,
        max_seq_len,
        attn_scale,
        stride_qm,
        stride_qh,
        stride_kn,
        stride_kh,
        stride_vn,
        stride_vh,
        stride_om,
        stride_oh,
        start_m,
        off_h,
        off_h_kv,
        seq_start_kv,
        seq_len_kv,
        seq_start_q,
        seq_len_q,
        M,
        stride_mm,
        uih_len_q,
        num_softmax_heads,
        HEAD_DIM,
        BLOCK_D_V,
        BLOCK_M,
        BLOCK_N,
        NUM_STAGES,
        SHARED_KV=SHARED_KV,
        HAS_CAUSAL=HAS_CAUSAL,
        LAZY_RESCALE=LAZY_RESCALE,
        RESCALE_THRESHOLD=RESCALE_THRESHOLD,
    )


@maybe_register_custom_op("tlx::hstu_cross_attn_gfx950_fwd", mutates_args=())
def tlx_gfx950_cross_attn_fwd(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_offsets_q: torch.Tensor,
    max_q_len: int,
    attn_scale: torch.Tensor,
    G: int,
    shared_kv: bool = False,
    num_softmax_heads: int = 0,
    enable_tma: bool = False,
    truncate_method: str = "none",
    num_targets: Optional[torch.Tensor] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert not enable_tma, "TMA is CUDA-only and not supported on gfx950"
    # Under `shared_kv` the kernel takes the PV operand from K's staged tile
    # and never reads `V`, so a `v` that is not `k` would be ignored rather
    # than rejected -- wrong output, no error. `AttentionFunction.forward`
    # substitutes `v = k`, but this op is registered and can be called
    # directly. Checked before the conversions below: they would turn a single
    # non-contiguous tensor passed as both into two separate copies, which
    # still computes the right answer but no longer compares equal here.
    assert not shared_kv or (v.data_ptr() == k.data_ptr() and v.shape == k.shape and v.stride()
                             == k.stride()), "shared_kv=True requires v to alias k; a distinct v would be ignored"
    q = switch_to_contiguous_if_needed(q)
    k = switch_to_contiguous_if_needed(k)
    v = switch_to_contiguous_if_needed(v)

    # Validate dtype consistency: q, k, v must have matching dtypes for
    # correct type casting in kernels (e.g., dv.to(q.dtype))
    assert q.dtype == k.dtype, f"q.dtype ({q.dtype}) != k.dtype ({k.dtype})"
    assert q.dtype == v.dtype, f"q.dtype ({q.dtype}) != v.dtype ({v.dtype})"
    # The silu epilogue reads the scale with a scalar `tl.load(attn_scale)`, so
    # a per-row tensor would silently apply element 0 to every row.
    assert attn_scale.ndim == 0, ("per-row attn_scale is not supported; pass a 0-dim tensor")

    Z = seq_offsets.numel() - 1
    total_seq_len_q, H, DimQ = q.shape
    _, _, DimV = v.shape
    out = torch.empty(total_seq_len_q, H, DimV, device=q.device, dtype=q.dtype)
    if total_seq_len_q == 0:
        M = torch.empty(0, device=q.device, dtype=torch.float32)
        return out, M

    grid = lambda meta: (  # noqa E731
        triton.cdiv(max_q_len, meta["BLOCK_M"]) * Z * H, )

    # Create M tensor for softmax heads if needed
    if num_softmax_heads > 0:
        M, stride_mm = forward_custom_vars(q, num_softmax_heads, [])
    else:
        M = torch.empty(0, device=q.device, dtype=torch.float32)
        stride_mm = 0

    num_cus = _num_cus(q.device.index if q.device.index is not None else 0)
    if os.environ.get(STREAM_K_ENV, "1") == "1" and stream_k_supported(q, k, v, Z, num_softmax_heads, shared_kv, causal,
                                                                       truncate_method, num_cus):
        stream_k_fwd(q, k, v, seq_offsets, seq_offsets_q, alpha, attn_scale, G, out,
                     M if num_softmax_heads > 0 else None, num_cus)
        return out, M

    _tlx_gfx950_cross_attn_fwd[grid](
        alpha=alpha,
        Z=Z,
        H=H,
        G=G,
        Q=q,
        K=k,
        V=v,
        Out=out,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_om=out.stride(0),
        stride_oh=out.stride(1),
        seq_offsets_q=seq_offsets_q,
        seq_offsets=seq_offsets,
        max_seq_len=max_seq_len,
        attn_scale=attn_scale,
        M=M,
        stride_mm=stride_mm,
        num_targets=num_targets,
        MAX_Q_LEN=max_q_len,
        AUTOTUNE_Z=next_power_of_2(Z),
        AUTOTUNE_MAX_Q_LEN=triton.cdiv(max_q_len, AUTOTUNE_Q_LEN_GRANULARITY) * AUTOTUNE_Q_LEN_GRANULARITY,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
        num_softmax_heads=num_softmax_heads,
        HEAD_DIM=DimQ,
        BLOCK_D_V=DimV,
        SHARED_KV=shared_kv,
        TRUNCATE_METHOD=truncate_method,
        HAS_CAUSAL=causal,
        HAS_NUM_TARGETS=num_targets is not None,
        # BF16 only: the packed-FP32 subtract hazard (module docstring) involves the FP16 MFMA. Training shares K/V,
        # so it keeps the exact rescale.
        LAZY_RESCALE=q.dtype == torch.bfloat16 and not shared_kv and not causal,
        RESCALE_THRESHOLD=LAZY_RESCALE_THRESHOLD.value,
    )
    return out, M


@triton.jit
def _compute_bwd_reduce_dq_offsets(
    H,
    G,
    BLOCK_N,
    seq_offsets_q,
    seq_offsets,
    max_seq_len,
    TRUNCATE_METHOD: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    off_h_kv = off_h // G
    start_n = tl.program_id(1) * BLOCK_N
    seq_start_kv = tl.load(seq_offsets + off_z).to(tl.int32)
    seq_end_kv = tl.load(seq_offsets + off_z + 1).to(tl.int32)
    seq_len_kv = (seq_end_kv - seq_start_kv).to(tl.int32)
    # Truncate to last max_seq_len KV tokens if actual seq len exceeds max_seq_len
    if TRUNCATE_METHOD != "none":
        truncated_len = tl.minimum(seq_len_kv, max_seq_len.to(tl.int32))
        if TRUNCATE_METHOD == "keep_last":
            seq_start_kv = seq_start_kv + (seq_len_kv - truncated_len)
        seq_len_kv = truncated_len
    seq_start_q = tl.load(seq_offsets_q + off_z).to(tl.int32)
    seq_end_q = tl.load(seq_offsets_q + off_z + 1).to(tl.int32)
    seq_len_q = (seq_end_q - seq_start_q).to(tl.int32)

    return (
        off_h,
        off_h_kv,
        start_n,
        seq_start_kv,
        seq_len_kv,
        seq_start_q,
        seq_len_q,
        off_z,
    )


def _bwd_pre_hook_redq(nargs):
    """
    The redq variant reduces DQ through memory:
    - DQ: many KV blocks accumulate into the same Q rows, so it must be zeroed.
    - DK, DV: each instance owns distinct K/V rows and uses tl.store, except
      under GQA without PER_KV_HEAD, where Q heads sharing a KV head collide.
    """
    if "DQ" in nargs:
        nargs["DQ"].zero_()
        if nargs["TRUNCATE_METHOD"] != "none":
            # When truncating KV, we need to zero out the entire dk/dv tensor
            nargs["DK"].zero_()
            if not nargs["SHARED_KV"]:
                nargs["DV"].zero_()
    # When G > 1 (GQA), multiple Q heads write to the same K/V head,
    # so dk/dv must be zero-initialized for atomic accumulation
    if nargs["G"] > 1 and not nargs.get("PER_KV_HEAD", False) and "DK" in nargs:
        nargs["DK"].zero_()
        if not nargs["SHARED_KV"]:
            nargs["DV"].zero_()


def _get_bw_redq_configs() -> List[triton.Config]:
    configs = []
    for M in [32, 64]:
        for N in [64, 128]:
            for waves_per_eu in [1, 2, 3]:
                configs.append(
                    triton.Config(
                        {
                            "BLOCK_M": M,
                            "BLOCK_N": N,
                            "matrix_instr_nonkdim": 16,
                            "waves_per_eu": waves_per_eu,
                        },
                        num_stages=1,
                        num_warps=4,
                        pre_hook=_bwd_pre_hook_redq,
                    ))
    return configs


def _bwd_smoke_configs() -> List[triton.Config]:
    return [
        c for c in _get_bw_redq_configs()
        if c.kwargs["BLOCK_M"] == 32 and c.kwargs["BLOCK_N"] == 64 and c.kwargs["waves_per_eu"] == 1
    ]


@triton.jit
def _bwd_load_2d(
    BASE,
    row_start,
    head_col,
    row_stride,
    row_end,
    BLOCK: tl.constexpr,
    DIM: tl.constexpr,
):
    """Load a [BLOCK, DIM] tile by pointer arithmetic.

    The backward kernels operate on tensors packed as [n_rows, H * DIM] (row =
    flattened sequence position, columns = heads concatenated along the last
    dim). ``head_col`` is the column offset of the current head within that
    packed layout (i.e. ``off_h * stride_head``). ``row_start`` is the absolute
    starting row and ``row_end`` is the absolute end row (exclusive) used to
    mask out-of-bounds rows.
    """
    offs_r = row_start + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, DIM)
    ptrs = (BASE + offs_r.to(tl.int64)[:, None] * row_stride.to(tl.int64) + (head_col + offs_d)[None, :])
    mask = (offs_r < row_end)[:, None]
    return tl.load(ptrs, mask=mask, other=0.0)


@triton.jit
def _tlx_gfx950_cross_attn_bwd_inner(  # noqa C901
    start_n,
    seq_start_kv,
    seq_len_kv,
    seq_start_q,
    seq_len_q,
    DV,
    stride_qh,
    stride_kh,
    stride_vh,
    stride_doh,
    stride_dvn,
    stride_dvh,
    alpha,
    attn_scale,
    M,
    Delta,
    stride_mm,
    off_h,
    off_h_kv,
    uih_len_q,
    num_softmax_heads: tl.constexpr,
    DimQ: tl.constexpr,
    DimV: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GQA_ATOMIC_ADD: tl.constexpr,
    SHARED_KV: tl.constexpr,
    MASK_KV: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    Q,
    K,
    V,
    DO,
    DQ,
    DK,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_dom,
    stride_dqm,
    stride_dkn,
):
    kv_offset = (seq_start_kv + start_n).to(tl.int32)
    end_kv = seq_start_kv + seq_len_kv
    end_dq = seq_start_q + seq_len_q
    k = _bwd_load_2d(
        K,
        kv_offset,
        off_h_kv * stride_kh,
        stride_kn,
        end_kv,
        BLOCK=BLOCK_N,
        DIM=DimQ,
    )
    if SHARED_KV:
        v = k
    else:
        v = _bwd_load_2d(
            V,
            kv_offset,
            off_h_kv * stride_vh,
            stride_vn,
            end_kv,
            BLOCK=BLOCK_N,
            DIM=DimV,
        )

    dk = tl.zeros([BLOCK_N, DimQ], dtype=tl.float32)
    if not SHARED_KV:
        dv = tl.zeros([BLOCK_N, DimV], dtype=tl.float32)
    # Offset DV pointers by sequence start and head offset
    if not SHARED_KV:
        DV = (DV + tl.cast(seq_start_kv, tl.int64) * tl.cast(stride_dvn, tl.int64) +
              off_h_kv * tl.cast(stride_dvh, tl.int64))
    scale = tl.load(attn_scale).to(tl.float32)

    M_off, Delta_off = backward_common_preprocess(M, Delta, off_h, num_softmax_heads, seq_start_q, stride_mm)

    offs_n = start_n + tl.arange(0, BLOCK_N)
    if off_h < num_softmax_heads:
        scaled_alpha = alpha * 1.44269504
    else:
        scaled_alpha = alpha
    if HAS_CAUSAL:
        low_q = max(0, start_n + uih_len_q - seq_len_kv)
    else:
        low_q = 0

    # Single-buffered LDS staging for the two tiles the q loop reloads every
    # iteration. k and v are loop-invariant and stay on `tl.load`: staging them
    # too would reach 64KB at BLOCK_M=64 / BLOCK_N=128 with DimQ=DimV=128,
    # which is where the autotuner would start trading block size for LDS.
    #
    # The transposes stay in registers. Both tiles are consumed in both
    # orientations -- q as `k @ q.T` and then `dqk.T @ q`, do as `act_qk.T @ do`
    # and then `v @ do.T` -- and serving both from one staged buffer via
    # `tlx.local_trans` fails in `TlxPropagateLayout`, because the two
    # `local_load`s want different MFMA operand layouts out of the same shared
    # allocation. The forward avoids this: each buffer there has a single
    # consumer orientation. So what staging buys the backward is the
    # global-to-LDS copy and the one-group overlap of the q and do loads, not
    # the transposes.
    # pyrefly: ignore [bad-argument-type]
    lds_q = tlx.local_alloc((BLOCK_M, DimQ), tlx.dtype_of(Q), 1)
    # pyrefly: ignore [bad-argument-type]
    lds_do = tlx.local_alloc((BLOCK_M, DimV), tlx.dtype_of(DO), 1)

    for start_m in tl.range(low_q, seq_len_q, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < seq_len_q
        q_offset = seq_start_q + start_m
        offs_r_m = q_offset + tl.arange(0, BLOCK_M)
        mask_r_m = (offs_r_m < end_dq)[:, None]
        q_ptrs = (Q + offs_r_m.to(tl.int64)[:, None] * stride_qm.to(tl.int64) +
                  (off_h * stride_qh + tl.arange(0, DimQ))[None, :])
        do_ptrs = (DO + offs_r_m.to(tl.int64)[:, None] * stride_dom.to(tl.int64) +
                   (off_h * stride_doh + tl.arange(0, DimV))[None, :])
        q_local = tlx.local_view(lds_q, 0)
        do_local = tlx.local_view(lds_do, 0)
        # One async group: do's address does not depend on the QK dot, so
        # issuing it here costs the iteration one global latency, not two.
        tok_q = tlx.async_load(q_ptrs, q_local, mask=mask_r_m)
        tok_do = tlx.async_load(do_ptrs, do_local, mask=mask_r_m)
        tlx.async_load_commit_group([tok_q, tok_do])
        # pyrefly: ignore [bad-argument-type]
        wait_tok = tlx.async_load_wait_group(0)
        q = tlx.local_load(q_local, token=wait_tok)
        qk_trans = tl.dot(
            k,
            tl.trans(q),
        )
        if MASK_KV or HAS_CAUSAL:
            valid_mask_trans = backward_valid_mask(
                offs_m,
                offs_n,
                uih_len_q,
                seq_len_q,
                seq_len_kv,
                HAS_CAUSAL,
            )
        else:
            valid_mask_trans = offs_m[None, :] < seq_len_q
        if off_h < num_softmax_heads:
            qk_trans, act_qk_trans, pT = backward_softmax_activation_scaled_alpha(
                qk_trans,
                scaled_alpha,
                valid_mask_trans,
                M_off,
                offs_m,
                stride_mm,
                mask_m,
                k,
            )
        else:
            qk_trans, act_qk_trans, pT = backward_silu_activation(qk_trans, alpha, valid_mask_trans, k.dtype, scale)
        do = tlx.local_load(do_local, token=wait_tok)

        if SHARED_KV:
            dk = tl.dot(act_qk_trans, do, dk, allow_tf32=ALLOW_TF32)
        else:
            # pyrefly: ignore [unbound-name]
            dv = tl.dot(act_qk_trans, do, dv, allow_tf32=ALLOW_TF32)

        dact_qk_trans = tl.dot(v, tl.trans(do), allow_tf32=ALLOW_TF32)
        if off_h < num_softmax_heads:
            dqk_trans = backward_d_softmax_activation(dact_qk_trans, Delta_off, offs_m, stride_mm, mask_m, pT)
        else:
            dqk_trans = backward_d_silu_activation(dact_qk_trans, pT, qk_trans, scale, valid_mask_trans)
        dqk_trans = dqk_trans.to(k.dtype)
        if SHARED_KV:
            dk_attn = tl.dot(dqk_trans, q, allow_tf32=ALLOW_TF32)
            dk = dk + dk_attn * alpha
        else:
            dk = tl.dot(dqk_trans, q, dk, allow_tf32=ALLOW_TF32)
        dq_trans = tl.dot(tl.trans(k), dqk_trans)
        dq_trans = dq_trans * alpha
        dq = tl.trans(dq_trans)
        offs_qk_d = tl.arange(0, DimQ)
        dq_ptrs = (DQ + (seq_start_q + offs_m).to(tl.int64)[:, None] * stride_dqm.to(tl.int64) +
                   (DimQ * off_h + offs_qk_d)[None, :])
        cur = tl.load(dq_ptrs, mask=mask_m[:, None], other=0.0)
        tl.store(
            dq_ptrs,
            cur + dq.to(DQ.dtype.element_ty),
            mask=mask_m[:, None],
        )

    offs_n = start_n + tl.arange(0, BLOCK_N)
    if not SHARED_KV:
        offs_v_d = tl.arange(0, DimV)
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_v_d[None, :])
        if MASK_KV:
            mask_n = offs_n < seq_len_kv
            if GQA_ATOMIC_ADD:
                tl.atomic_add(
                    dv_ptrs,
                    # pyrefly: ignore [unbound-name]
                    dv.to(k.dtype),
                    mask=mask_n[:, None],
                    sem="relaxed",
                )
            else:
                tl.store(
                    dv_ptrs,
                    # pyrefly: ignore [unbound-name]
                    dv.to(k.dtype),
                    mask=mask_n[:, None],
                )
        else:
            if GQA_ATOMIC_ADD:
                tl.atomic_add(
                    dv_ptrs,
                    # pyrefly: ignore [unbound-name]
                    dv.to(k.dtype),
                    sem="relaxed",
                )
            else:
                tl.store(
                    dv_ptrs,
                    # pyrefly: ignore [unbound-name]
                    dv.to(k.dtype),
                )

    if not SHARED_KV:
        dk = dk * alpha

    offs_qk_d = tl.arange(0, DimQ)
    dk_ptrs = (DK + (kv_offset + tl.arange(0, BLOCK_N)).to(tl.int64)[:, None] * stride_dkn.to(tl.int64) +
               (DimQ * off_h_kv + offs_qk_d)[None, :])
    mask_n = offs_n < seq_len_kv
    if GQA_ATOMIC_ADD:
        tl.atomic_add(
            dk_ptrs,
            dk.to(DK.dtype.element_ty),
            mask=mask_n[:, None],
            sem="relaxed",
        )
    else:
        tl.store(dk_ptrs, dk.to(DK.dtype.element_ty), mask=mask_n[:, None])


@_space_autotune(
    configs=_get_bw_redq_configs(),
    smoke_configs=_bwd_smoke_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "max_q_len",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
        "num_softmax_heads",
    ],
)
@triton.jit
def _tlx_gfx950_cross_attn_bwd(  # noqa C901
    Q,
    K,
    V,
    DO,
    seq_offsets,
    seq_offsets_q,
    DQ,
    DK,
    DV,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_dom,
    stride_doh,
    stride_dqm,
    stride_dqh,
    stride_dkn,
    stride_dkh,
    stride_dvn,
    stride_dvh,
    alpha,
    max_seq_len,
    attn_scale,
    M,
    Delta,
    stride_mm,
    num_targets,
    Z,
    AUTOTUNE_Z,
    H,
    G: tl.constexpr,
    num_softmax_heads: tl.constexpr,
    max_q_len: tl.constexpr,
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ: tl.constexpr,
    DimV: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SHARED_KV: tl.constexpr,
    TRUNCATE_METHOD: tl.constexpr,
    HAS_CAUSAL: tl.constexpr,
    HAS_NUM_TARGETS: tl.constexpr,
    # pyrefly: ignore [bad-function-definition]
    PER_KV_HEAD: tl.constexpr = False,
):
    if PER_KV_HEAD:
        # Per-KV-head (atomic-free GQA) variant: one program owns a KV head and
        # walks its G query heads, so dk/dv reduce in registers.
        H_kv = H // G
        off_hz = tl.program_id(0)
        off_z = off_hz // H_kv
        off_h_kv = off_hz % H_kv
        seq_start_kv = tl.load(seq_offsets + off_z).to(tl.int32)
        seq_end_kv = tl.load(seq_offsets + off_z + 1).to(tl.int32)
        seq_len_kv = (seq_end_kv - seq_start_kv).to(tl.int32)
        if TRUNCATE_METHOD != "none":
            truncated_len = tl.minimum(seq_len_kv, max_seq_len.to(tl.int32))
            if TRUNCATE_METHOD == "keep_last":
                seq_start_kv = seq_start_kv + (seq_len_kv - truncated_len)
            seq_len_kv = truncated_len
        seq_start_q = tl.load(seq_offsets_q + off_z).to(tl.int32)
        seq_end_q = tl.load(seq_offsets_q + off_z + 1).to(tl.int32)
        seq_len_q = (seq_end_q - seq_start_q).to(tl.int32)
        if HAS_CAUSAL:
            n_targets = target_common_preprocess(off_z, num_targets, HAS_NUM_TARGETS)
            uih_len_q = uih_common_preprocess(n_targets, seq_len_q, HAS_NUM_TARGETS)
        else:
            uih_len_q = seq_len_q
        if seq_len_kv == 0 or seq_len_q == 0:
            return

        end_dq = seq_start_q + seq_len_q
        offs_qk_d = tl.arange(0, DimQ)
        offs_v_d = tl.arange(0, DimV)
        scale = tl.load(attn_scale).to(tl.float32)

        # Same staging as `_tlx_gfx950_cross_attn_bwd_inner`: the q loop reloads
        # q and do every iteration, so both are copied into LDS once per
        # iteration and issued as a single async group. k and v are
        # loop-invariant and stay on `tl.load`.
        # pyrefly: ignore [bad-argument-type]
        lds_q = tlx.local_alloc((BLOCK_M, DimQ), tlx.dtype_of(Q), 1)
        # pyrefly: ignore [bad-argument-type]
        lds_do = tlx.local_alloc((BLOCK_M, DimV), tlx.dtype_of(DO), 1)

        # pyrefly: ignore [bad-argument-type]
        for start_n in tl.range(0, seq_len_kv, BLOCK_N, num_stages=1):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < seq_len_kv
            k_offset = (seq_start_kv + start_n).to(tl.int32)
            k = _bwd_load_2d(
                K,
                k_offset,
                off_h_kv * stride_kh,
                stride_kn,
                seq_start_kv + seq_len_kv,
                BLOCK=BLOCK_N,
                DIM=DimQ,
            )
            if SHARED_KV:
                v = k
            else:
                v = _bwd_load_2d(
                    V,
                    k_offset,
                    off_h_kv * stride_vh,
                    stride_vn,
                    seq_start_kv + seq_len_kv,
                    BLOCK=BLOCK_N,
                    DIM=DimV,
                )
            dk = tl.zeros([BLOCK_N, DimQ], dtype=tl.float32)
            if not SHARED_KV:
                dv = tl.zeros([BLOCK_N, DimV], dtype=tl.float32)
            for g in range(G):
                off_h = off_h_kv * G + g
                if off_h < num_softmax_heads:
                    scaled_alpha = alpha * 1.44269504
                else:
                    scaled_alpha = alpha
                M_off, Delta_off = backward_common_preprocess(M, Delta, off_h, num_softmax_heads, seq_start_q,
                                                              stride_mm)
                # pyrefly: ignore [bad-argument-type]
                for start_m in tl.range(0, seq_len_q, BLOCK_M, num_stages=1):
                    offs_m = start_m + tl.arange(0, BLOCK_M)
                    mask_m = offs_m < seq_len_q
                    valid_mask_trans = backward_valid_mask(offs_m, offs_n, uih_len_q, seq_len_q, seq_len_kv, HAS_CAUSAL)
                    q_offset = (seq_start_q + start_m).to(tl.int32)
                    offs_r_m = q_offset + tl.arange(0, BLOCK_M)
                    mask_r_m = (offs_r_m < end_dq)[:, None]
                    q_ptrs = (Q + offs_r_m.to(tl.int64)[:, None] * stride_qm.to(tl.int64) +
                              (off_h * stride_qh + offs_qk_d)[None, :])
                    do_ptrs = (DO + offs_r_m.to(tl.int64)[:, None] * stride_dom.to(tl.int64) +
                               (off_h * stride_doh + offs_v_d)[None, :])
                    q_local = tlx.local_view(lds_q, 0)
                    do_local = tlx.local_view(lds_do, 0)
                    tok_q = tlx.async_load(q_ptrs, q_local, mask=mask_r_m)
                    tok_do = tlx.async_load(do_ptrs, do_local, mask=mask_r_m)
                    tlx.async_load_commit_group([tok_q, tok_do])
                    # pyrefly: ignore [bad-argument-type]
                    wait_tok = tlx.async_load_wait_group(0)
                    q = tlx.local_load(q_local, token=wait_tok)
                    do = tlx.local_load(do_local, token=wait_tok)
                    qk_trans = tl.dot(k, tl.trans(q), allow_tf32=ALLOW_TF32)
                    if off_h < num_softmax_heads:
                        qk_trans, act_qk_trans, pT = (backward_softmax_activation_scaled_alpha(
                            qk_trans,
                            scaled_alpha,
                            valid_mask_trans,
                            M_off,
                            offs_m,
                            stride_mm,
                            mask_m,
                            k,
                        ))
                    else:
                        qk_trans, act_qk_trans, pT = backward_silu_activation(qk_trans, alpha, valid_mask_trans,
                                                                              k.dtype, scale)
                    if SHARED_KV:
                        dk += tl.dot(act_qk_trans, do, allow_tf32=ALLOW_TF32)
                    else:
                        # pyrefly: ignore [unbound-name]
                        dv += tl.dot(act_qk_trans, do, allow_tf32=ALLOW_TF32)
                    dact_qk_trans = tl.dot(v, tl.trans(do), allow_tf32=ALLOW_TF32)
                    if off_h < num_softmax_heads:
                        dqk_trans = backward_d_softmax_activation(dact_qk_trans, Delta_off, offs_m, stride_mm, mask_m,
                                                                  pT)
                    else:
                        dqk_trans = backward_d_silu_activation(dact_qk_trans, pT, qk_trans, scale, valid_mask_trans)
                    dqk_trans = dqk_trans.to(k.dtype)
                    dk += tl.dot(dqk_trans, q, allow_tf32=ALLOW_TF32) * alpha
                    dq_trans = (tl.dot(tl.trans(k), dqk_trans, allow_tf32=ALLOW_TF32) * alpha)
                    dq = tl.trans(dq_trans)
                    offs_d = tl.arange(0, DimQ)
                    dq_ptrs = (DQ + (seq_start_q + offs_m).to(tl.int64)[:, None] * stride_dqm.to(tl.int64) +
                               (DimQ * off_h + offs_d)[None, :])
                    cur = tl.load(dq_ptrs, mask=mask_m[:, None], other=0.0)
                    tl.store(
                        dq_ptrs,
                        cur + dq.to(DQ.dtype.element_ty),
                        mask=mask_m[:, None],
                    )
            DK_off = (DK + tl.cast(seq_start_kv, tl.int64) * tl.cast(stride_dkn, tl.int64) +
                      off_h_kv * tl.cast(stride_dkh, tl.int64))
            dk_ptrs = DK_off + (offs_n[:, None] * stride_dkn + offs_qk_d[None, :])
            tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])
            if not SHARED_KV:
                DV_off = (DV + tl.cast(seq_start_kv, tl.int64) * tl.cast(stride_dvn, tl.int64) +
                          off_h_kv * tl.cast(stride_dvh, tl.int64))
                dv_ptrs = DV_off + (offs_n[:, None] * stride_dvn + offs_v_d[None, :])
                # pyrefly: ignore [unbound-name]
                tl.store(dv_ptrs, dv.to(k.dtype), mask=mask_n[:, None])
        return
    (
        off_h,
        off_h_kv,
        start_n,
        seq_start_kv,
        seq_len_kv,
        seq_start_q,
        seq_len_q,
        off_z,
    ) = _compute_bwd_reduce_dq_offsets(
        H,
        G,
        BLOCK_N,
        seq_offsets_q,
        seq_offsets,
        max_seq_len,
        TRUNCATE_METHOD,
    )
    n_targets = target_common_preprocess(off_z, num_targets, HAS_NUM_TARGETS)
    uih_len_q = uih_common_preprocess(n_targets, seq_len_q, HAS_NUM_TARGETS)
    if seq_len_kv == 0:
        return
    last_block_start_n = tl.maximum(0, (tl.cdiv(seq_len_kv, BLOCK_N) - 1) * BLOCK_N)
    for start_n in tl.range(0, last_block_start_n, BLOCK_N):
        _tlx_gfx950_cross_attn_bwd_inner(
            start_n,
            seq_start_kv,
            seq_len_kv,
            seq_start_q,
            seq_len_q,
            DV,
            stride_qh,
            stride_kh,
            stride_vh,
            stride_doh,
            stride_dvn,
            stride_dvh,
            alpha,
            attn_scale,
            M,
            Delta,
            stride_mm,
            off_h,
            off_h_kv,
            uih_len_q,
            num_softmax_heads,
            DimQ,
            DimV,
            ALLOW_TF32,
            BLOCK_M,
            BLOCK_N,
            GQA_ATOMIC_ADD=G > 1,
            SHARED_KV=SHARED_KV,
            # pyrefly: ignore [bad-argument-type]
            MASK_KV=False,
            HAS_CAUSAL=HAS_CAUSAL,
            Q=Q,
            K=K,
            V=V,
            DO=DO,
            DQ=DQ,
            DK=DK,
            stride_qm=stride_qm,
            stride_kn=stride_kn,
            stride_vn=stride_vn,
            stride_dom=stride_dom,
            stride_dqm=stride_dqm,
            stride_dkn=stride_dkn,
        )
    # Handle last block with masking
    _tlx_gfx950_cross_attn_bwd_inner(
        last_block_start_n,
        seq_start_kv,
        seq_len_kv,
        seq_start_q,
        seq_len_q,
        DV,
        stride_qh,
        stride_kh,
        stride_vh,
        stride_doh,
        stride_dvn,
        stride_dvh,
        alpha,
        attn_scale,
        M,
        Delta,
        stride_mm,
        off_h,
        off_h_kv,
        uih_len_q,
        num_softmax_heads,
        DimQ,
        DimV,
        ALLOW_TF32,
        BLOCK_M,
        BLOCK_N,
        GQA_ATOMIC_ADD=G > 1,
        SHARED_KV=SHARED_KV,
        # pyrefly: ignore [bad-argument-type]
        MASK_KV=True,
        HAS_CAUSAL=HAS_CAUSAL,
        Q=Q,
        K=K,
        V=V,
        DO=DO,
        DQ=DQ,
        DK=DK,
        stride_qm=stride_qm,
        stride_kn=stride_kn,
        stride_vn=stride_vn,
        stride_dom=stride_dom,
        stride_dqm=stride_dqm,
        stride_dkn=stride_dkn,
    )


@triton.jit
def _tlx_gfx950_cross_attn_bwd_split_softmax_dkdv(  # noqa: TR001
    Q,
    K,
    DOut,
    DK,
    seq_offsets,
    seq_offsets_q,
    M,
    Delta,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_dom,
    stride_doh,
    stride_dkn,
    stride_dkh,
    stride_mm,
    alpha,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = (off_hz % H).to(tl.int64)
    start_n = tl.program_id(1) * BLOCK_N
    seq_start_kv = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end_kv = tl.load(seq_offsets + off_z + 1)
    seq_len_kv = (seq_end_kv - seq_start_kv).to(tl.int32)
    if start_n >= seq_len_kv:
        return
    seq_start_q = tl.load(seq_offsets_q + off_z).to(tl.int64)
    seq_end_q = tl.load(seq_offsets_q + off_z + 1)
    seq_len_q = (seq_end_q - seq_start_q).to(tl.int32)

    Q += seq_start_q * stride_qm + off_h * stride_qh
    K += seq_start_kv * stride_kn + off_h * stride_kh
    DOut += seq_start_q * stride_dom + off_h * stride_doh
    DK += seq_start_kv * stride_dkn + off_h * stride_dkh

    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n < seq_len_kv
    k = tl.load(
        K + offs_n[:, None] * stride_kn + offs_d[None, :],
        mask=mask_n[:, None] & (offs_d[None, :] < BLOCK_D),
        other=0.0,
    )
    dk = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)
    scaled_alpha = alpha * 1.44269504
    offs_m_base = tl.arange(0, BLOCK_M)
    for start_m in tl.range(0, seq_len_q, BLOCK_M):
        offs_m = start_m + offs_m_base
        mask_m = offs_m < seq_len_q
        q = tl.load(
            Q + offs_m[:, None] * stride_qm + offs_d[None, :],
            mask=mask_m[:, None] & (offs_d[None, :] < BLOCK_D),
            other=0.0,
        )
        q_trans = tl.trans(q)
        do = tl.load(
            DOut + offs_m[:, None] * stride_dom + offs_d[None, :],
            mask=mask_m[:, None] & (offs_d[None, :] < BLOCK_D),
            other=0.0,
        )
        do_trans = tl.trans(do)
        valid = mask_n[:, None] & mask_m[None, :]
        qk_trans = tl.dot(k, q_trans, allow_tf32=ALLOW_TF32)
        m = tl.load(
            M + seq_start_q * stride_mm + off_h + offs_m * stride_mm,
            mask=mask_m,
        )
        p = tl.math.exp2(qk_trans * scaled_alpha - m[None, :])
        p = tl.where(valid, p, 0.0)
        p_bf16 = p.to(k.dtype)
        dk += tl.dot(p_bf16, do, allow_tf32=ALLOW_TF32)
        dact = tl.dot(k, do_trans, allow_tf32=ALLOW_TF32)
        delta = tl.load(
            Delta + seq_start_q * stride_mm + off_h + offs_m * stride_mm,
            mask=mask_m,
        )
        dqk = (p * (dact - delta[None, :])).to(k.dtype)
        dk += tl.dot(dqk, q, allow_tf32=ALLOW_TF32) * alpha

    tl.store(
        DK + offs_n[:, None] * stride_dkn + offs_d[None, :],
        dk.to(k.dtype),
        mask=mask_n[:, None] & (offs_d[None, :] < BLOCK_D),
    )


@triton.jit
def _tlx_gfx950_cross_attn_bwd_split_softmax_dq(  # noqa: TR001
    Q,
    K,
    DOut,
    DQ,
    seq_offsets,
    seq_offsets_q,
    M,
    Delta,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_dom,
    stride_doh,
    stride_dqm,
    stride_dqh,
    stride_mm,
    alpha,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = (off_hz % H).to(tl.int64)
    start_m = tl.program_id(1) * BLOCK_M
    seq_start_kv = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end_kv = tl.load(seq_offsets + off_z + 1)
    seq_len_kv = (seq_end_kv - seq_start_kv).to(tl.int32)
    seq_start_q = tl.load(seq_offsets_q + off_z).to(tl.int64)
    seq_end_q = tl.load(seq_offsets_q + off_z + 1)
    seq_len_q = (seq_end_q - seq_start_q).to(tl.int32)
    if start_m >= seq_len_q:
        return

    Q += seq_start_q * stride_qm + off_h * stride_qh
    K += seq_start_kv * stride_kn + off_h * stride_kh
    DOut += seq_start_q * stride_dom + off_h * stride_doh
    DQ += seq_start_q * stride_dqm + off_h * stride_dqh

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < seq_len_q
    q = tl.load(
        Q + offs_m[:, None] * stride_qm + offs_d[None, :],
        mask=mask_m[:, None] & (offs_d[None, :] < BLOCK_D),
        other=0.0,
    )
    do = tl.load(
        DOut + offs_m[:, None] * stride_dom + offs_d[None, :],
        mask=mask_m[:, None] & (offs_d[None, :] < BLOCK_D),
        other=0.0,
    )
    m = tl.load(
        M + seq_start_q * stride_mm + off_h + offs_m * stride_mm,
        mask=mask_m,
    )
    delta = tl.load(
        Delta + seq_start_q * stride_mm + off_h + offs_m * stride_mm,
        mask=mask_m,
    )
    dq = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)
    scaled_alpha = alpha * 1.44269504
    offs_n_base = tl.arange(0, BLOCK_N)
    for start_n in tl.range(0, seq_len_kv, BLOCK_N):
        offs_n = start_n + offs_n_base
        mask_n = offs_n < seq_len_kv
        k = tl.load(
            K + offs_n[:, None] * stride_kn + offs_d[None, :],
            mask=mask_n[:, None] & (offs_d[None, :] < BLOCK_D),
            other=0.0,
        )
        k_trans = tl.trans(k)
        valid = mask_m[:, None] & mask_n[None, :]
        qk = tl.dot(q, k_trans, allow_tf32=ALLOW_TF32)
        p = tl.math.exp2(qk * scaled_alpha - m[:, None])
        p = tl.where(valid, p, 0.0)
        dact = tl.dot(do, k_trans, allow_tf32=ALLOW_TF32)
        dqk = (p * (dact - delta[:, None])).to(k.dtype)
        dq += tl.dot(dqk, k, allow_tf32=ALLOW_TF32) * alpha

    tl.store(
        DQ + offs_m[:, None] * stride_dqm + offs_d[None, :],
        dq.to(q.dtype),
        mask=mask_m[:, None] & (offs_d[None, :] < BLOCK_D),
    )


def _validate_backward_variant(bwd_variant, truncate_method):
    torch._assert(
        bwd_variant in (None, "redq", "split_softmax", "v3_ttgir", "fp32_pipeline_qdo", "v3_ttgir_fp32_pipeline_qdo"),
        "Unsupported HSTU cross-attention backward variant",
    )
    if bwd_variant not in (None, "redq", "split_softmax"):
        torch._assert(truncate_method == "none", "Pipeline backward does not support truncation")


def _tlx_gfx950_cross_attn_bwd_impl(  # noqa C901
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    max_seq_len: int,
    alpha: float,
    max_q_len: Optional[int],
    seq_offsets_q: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    causal: bool,
    shared_kv: bool,
    G: int,
    num_softmax_heads: int = 0,
    M: Optional[torch.Tensor] = None,
    Delta: Optional[torch.Tensor] = None,
    stride_mm: int = 0,
    truncate_method: str = "none",
    enable_tma: bool = False,
    bwd_variant: Optional[str] = None,
    OUT: Optional[torch.Tensor] = None,
    backward_route: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert not enable_tma, "TMA is CUDA-only and not supported on gfx950"
    _validate_backward_variant(bwd_variant, truncate_method)
    if max_q_len is None:
        max_q_len = max_seq_len
        assert seq_offsets_q is None
        seq_offsets_q = seq_offsets

    q = switch_to_contiguous_if_needed(q)
    k = switch_to_contiguous_if_needed(k)
    v = k if shared_kv else switch_to_contiguous_if_needed(v)
    if num_targets is not None:
        num_targets = num_targets.contiguous()
    dout = dout.contiguous()

    # Validate dtype consistency: q, k, v must have matching dtypes for
    # correct type casting in kernels (e.g., dk.to(q.dtype))
    assert q.dtype == k.dtype, f"q.dtype ({q.dtype}) != k.dtype ({k.dtype})"
    assert q.dtype == v.dtype, f"q.dtype ({q.dtype}) != v.dtype ({v.dtype})"
    # Both backward paths read the scale with a scalar `tl.load`, matching the
    # forward.
    assert attn_scale.ndim == 0, ("per-row attn_scale is not supported; pass a 0-dim tensor")

    if dout.shape[0] == 0:
        dv = torch.empty(0, dtype=v.dtype, device=v.device) if shared_kv else torch.zeros_like(v)
        return torch.zeros_like(q), torch.zeros_like(k), dv
    Z = seq_offsets.numel() - 1
    total_seq_len_q, H, DimQ = q.shape
    _, _, DimV = v.shape

    # Pointer loads read DO using q's folded row layout. In the GQA fold,
    # forward saves q as [total * G, H_kv, D] but returns out as [total, H, D],
    # so reshape dout back to the saved-q layout before deriving DO strides.
    dout = dout.reshape(total_seq_len_q, H, dout.shape[-1])

    seq_offsets = seq_offsets.contiguous()
    seq_offsets_q = seq_offsets_q.contiguous()
    if backward_route is not None:
        assert backward_route in ("retained", "coarse", "coarse_cached", "compact", "small")
        # Saved-tensor hooks can return equal values with different strides.
        q = q.contiguous()
        k = k.contiguous()
        v = k if shared_kv else v.contiguous()
        eligible = (G == 1 and max_q_len == 256 and H == k.shape[1] == v.shape[1] == num_softmax_heads
                    and DimQ == DimV == 128 and not causal and not enable_tma and truncate_method == "none"
                    and q.dtype == k.dtype == v.dtype == dout.dtype == torch.bfloat16 and M is not None
                    and OUT is not None and seq_offsets.dtype == seq_offsets_q.dtype
                    and seq_offsets.dtype in (torch.int32, torch.int64)
                    and (backward_route != "retained" or seq_offsets.dtype == torch.int64)
                    and seq_offsets.ndim == seq_offsets_q.ndim == 1 and seq_offsets.numel() == seq_offsets_q.numel())
        if eligible:
            M = M.contiguous()
            OUT = OUT.contiguous()
            if backward_route == "retained":
                assert shared_kv and H == 1 and Z >= 256 and max_seq_len in (256, 512, 1024, 2048)
                dq_fast, dk_fast = retained_softmax_backward(q, k, dout, M, OUT, seq_offsets_q, seq_offsets, alpha)
                dv_fast = None
            elif backward_route in ("coarse", "coarse_cached"):
                assert shared_kv and H == 1 and 4 < Z <= 64 and 384 <= max_seq_len <= 2048
                dq_fast, dk_fast, dv_fast = coarse_softmax_backward(q, k, dout, M, OUT, seq_offsets_q, seq_offsets,
                                                                    alpha,
                                                                    cache_delta=backward_route == "coarse_cached")
            elif backward_route == "compact":
                assert 0 < Z <= 4 and 0 < max_seq_len <= 2048
                dq_fast, dk_fast, dv_fast = compact_softmax_backward(q, k, v, dout, M, OUT, seq_offsets_q, seq_offsets,
                                                                     max_seq_len, alpha, shared_kv)
            else:
                assert 0 < Z <= 64 and 0 < max_seq_len <= 2048
                dq_fast, dk_fast, dv_fast = small_softmax_backward(q, k, v, dout, M, OUT, seq_offsets_q, seq_offsets,
                                                                   max_seq_len, alpha, shared_kv)
            if shared_kv:
                dv_fast = torch.empty(0, dtype=v.dtype, device=v.device)
            return dq_fast, dk_fast, dv_fast

    if num_softmax_heads > 0:
        assert M is not None
        M = M.contiguous()
        stride_mm = M.stride(0)
        if Delta is None:
            assert OUT is not None, "Softmax backward requires Delta or the saved output"
            OUT = OUT.contiguous()
            Delta = torch.empty_like(M)
            pre_grid = (triton.cdiv(OUT.shape[0], 128), num_softmax_heads)
            _attn_bwd_preprocess[pre_grid](OUT, dout, Delta, OUT.shape[0], H=OUT.shape[1],
                                           softmax_heads=num_softmax_heads, BLOCK_M=128, HEAD_DIM=OUT.shape[2])
        else:
            Delta = Delta.contiguous()
    else:
        # The SiLU fallback still forms pointer expressions for both placeholders.
        if M is None:
            M = torch.empty(0, dtype=torch.float32, device=q.device)
        if Delta is None:
            Delta = torch.empty(0, dtype=torch.float32, device=q.device)

    if bwd_variant == "fp32_pipeline_qdo":
        bwd_variant = "v3_ttgir_fp32_pipeline_qdo"

    if bwd_variant == "split_softmax":
        torch._assert(G == 1, "Split-softmax requires G=1")
        torch._assert(H == 1, "Split-softmax requires one attention head")
        torch._assert(
            DimQ == 128 and DimV == 128,
            "Split-softmax requires D=128",
        )
        torch._assert(
            num_softmax_heads == H,
            "Split-softmax requires all heads to use softmax",
        )
        torch._assert(not causal, "Split-softmax is non-causal")
        torch._assert(
            attn_scale.ndim == 0,
            "Split-softmax requires scalar attention scale",
        )
        torch._assert(shared_kv, "Split-softmax requires shared K/V")
        torch._assert(
            seq_offsets_q is not None,
            "Split-softmax requires query offsets",
        )
        dq_softmax = torch.empty_like(q)
        dk_softmax = torch.empty_like(k)
        _tlx_gfx950_cross_attn_bwd_split_softmax_dkdv[(Z * H, triton.cdiv(max_seq_len, SPLIT_SOFTMAX_DKDV_BLOCK_N))](
            Q=q,
            K=k,
            DOut=dout,
            DK=dk_softmax,
            seq_offsets=seq_offsets,
            seq_offsets_q=seq_offsets_q,
            M=M,
            Delta=Delta,
            stride_qm=q.stride(0),
            stride_qh=q.stride(1),
            stride_kn=k.stride(0),
            stride_kh=k.stride(1),
            stride_dom=dout.stride(0),
            stride_doh=dout.stride(1),
            stride_dkn=dk_softmax.stride(0),
            stride_dkh=dk_softmax.stride(1),
            stride_mm=stride_mm,
            alpha=alpha,
            # pyrefly: ignore [bad-argument-type]
            H=H,
            # pyrefly: ignore [bad-argument-type]
            BLOCK_M=SPLIT_SOFTMAX_DKDV_BLOCK_M,
            # pyrefly: ignore [bad-argument-type]
            BLOCK_N=SPLIT_SOFTMAX_DKDV_BLOCK_N,
            # pyrefly: ignore [bad-argument-type]
            BLOCK_D=DimQ,
            # pyrefly: ignore [bad-argument-type]
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            # pyrefly: ignore [unexpected-keyword]
            num_warps=SPLIT_SOFTMAX_DKDV_NUM_WARPS,
            # pyrefly: ignore [unexpected-keyword]
            num_stages=SPLIT_SOFTMAX_DKDV_NUM_STAGES,
            # pyrefly: ignore [unexpected-keyword]
            matrix_instr_nonkdim=SPLIT_SOFTMAX_DKDV_MATRIX_INSTR_NONKDIM,
            # pyrefly: ignore [unexpected-keyword]
            waves_per_eu=SPLIT_SOFTMAX_DKDV_WAVES_PER_EU,
        )
        _tlx_gfx950_cross_attn_bwd_split_softmax_dq[(Z * H, triton.cdiv(max_q_len, SPLIT_SOFTMAX_DQ_BLOCK_M))](
            Q=q,
            K=k,
            DOut=dout,
            DQ=dq_softmax,
            seq_offsets=seq_offsets,
            seq_offsets_q=seq_offsets_q,
            M=M,
            Delta=Delta,
            stride_qm=q.stride(0),
            stride_qh=q.stride(1),
            stride_kn=k.stride(0),
            stride_kh=k.stride(1),
            stride_dom=dout.stride(0),
            stride_doh=dout.stride(1),
            stride_dqm=dq_softmax.stride(0),
            stride_dqh=dq_softmax.stride(1),
            stride_mm=stride_mm,
            alpha=alpha,
            # pyrefly: ignore [bad-argument-type]
            H=H,
            # pyrefly: ignore [bad-argument-type]
            BLOCK_M=SPLIT_SOFTMAX_DQ_BLOCK_M,
            # pyrefly: ignore [bad-argument-type]
            BLOCK_N=SPLIT_SOFTMAX_DQ_BLOCK_N,
            # pyrefly: ignore [bad-argument-type]
            BLOCK_D=DimQ,
            # pyrefly: ignore [bad-argument-type]
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            # pyrefly: ignore [unexpected-keyword]
            num_warps=SPLIT_SOFTMAX_DQ_NUM_WARPS,
            # pyrefly: ignore [unexpected-keyword]
            num_stages=SPLIT_SOFTMAX_DQ_NUM_STAGES,
            # pyrefly: ignore [unexpected-keyword]
            matrix_instr_nonkdim=SPLIT_SOFTMAX_DQ_MATRIX_INSTR_NONKDIM,
            # pyrefly: ignore [unexpected-keyword]
            waves_per_eu=SPLIT_SOFTMAX_DQ_WAVES_PER_EU,
        )
        return (
            dq_softmax,
            dk_softmax,
            torch.empty(0, dtype=v.dtype, device=v.device),
        )

    if bwd_variant not in (None, "redq"):
        torch._assert(
            seq_offsets.dtype == seq_offsets_q.dtype == torch.int64,
            "FP32 pipeline requires int64 sequence offsets",
        )
        torch._assert(G == 1, "V3 TTGIR baseline requires G=1")
        torch._assert(
            H == 1 and k.shape[1] == 1 and v.shape[1] == 1,
            "V3 TTGIR baseline requires one Q/K/V head",
        )
        torch._assert(
            DimQ == 128 and DimV == 128,
            "V3 TTGIR baseline requires D=128",
        )
        torch._assert(
            q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
            and dout.dtype == torch.bfloat16,
            "V3 TTGIR baseline requires BF16 tensors",
        )
        torch._assert(max_q_len == 256, "V3 TTGIR baseline requires max Q=256")
        torch._assert(
            num_softmax_heads == 1,
            "V3 TTGIR baseline requires one softmax head",
        )
        torch._assert(not causal, "V3 TTGIR baseline is non-causal")
        torch._assert(
            attn_scale.ndim == 0,
            "V3 TTGIR baseline requires scalar attention scale",
        )
        torch._assert(shared_kv, "V3 TTGIR baseline requires shared K/V")
        torch._assert(
            k.data_ptr() == v.data_ptr(),
            "V3 TTGIR baseline requires aliased K/V storage",
        )
        torch._assert(
            num_targets is not None,
            "V3 TTGIR baseline requires target counts",
        )
        torch._assert(
            seq_offsets_q is not None,
            "V3 TTGIR baseline requires query offsets",
        )
        single_kv_tile = bwd_variant == "v3_ttgir_fp32_pipeline_qdo" and max_seq_len == 128
        if single_kv_tile:
            # The single-tile path writes final dQ without FP32 scratch.
            dq_v3 = torch.empty_like(q)
        elif bwd_variant == "v3_ttgir_fp32_pipeline_qdo":
            dq_v3 = torch.empty_like(q, dtype=torch.float32)
        else:
            dq_v3 = torch.zeros_like(q, dtype=torch.float32)
        dq_final = (dq_v3 if single_kv_tile else
                    torch.empty_like(dq_v3, dtype=q.dtype) if bwd_variant == "v3_ttgir_fp32_pipeline_qdo" else None)
        dk_v3 = torch.empty_like(k)
        dv_v3 = torch.empty((0, v.shape[1], v.shape[2]), dtype=v.dtype, device=v.device)
        _tlx_gfx950_cross_attn_v3_ttgir_bwd[(Z * H, 1)](
            q,
            k,
            dout,
            q.shape[0],
            seq_offsets,
            seq_offsets_q,
            dq_v3,
            dk_v3,
            q.stride(0),
            k.stride(0),
            dout.stride(0),
            dq_v3.stride(0),
            dk_v3.stride(0),
            alpha,
            M,
            Delta,
            # pyrefly: ignore [bad-argument-type]
            PIPELINE_QDO=bwd_variant == "v3_ttgir_fp32_pipeline_qdo",
            DQ_FINAL=dq_final,
            EARLY_TAIL_QDO=(bwd_variant == "v3_ttgir_fp32_pipeline_qdo" and max_seq_len <= 128),
            SINGLE_KV_TILE=single_kv_tile,
            # Packed Q lengths are at most 256, so equality proves every length is 256.
            PACKED_FIXED_Q=(single_kv_tile or (bwd_variant == "v3_ttgir_fp32_pipeline_qdo" and max_seq_len == 384))
            and total_seq_len_q == Z * 256,
            FIXED_KV=(384 if bwd_variant == "v3_ttgir_fp32_pipeline_qdo" and max_seq_len == 384
                      and total_seq_len_q == Z * 256 and k.shape[0] == Z * 384 else 0),
            NATIVE_Q_SCORE=(bwd_variant == "v3_ttgir_fp32_pipeline_qdo" and max_seq_len == 384
                            and total_seq_len_q == Z * 256),
            CACHE_K=(bwd_variant == "v3_ttgir_fp32_pipeline_qdo" and max_seq_len >= 1024),
            # pyrefly: ignore [unexpected-keyword]
            num_warps=4,
            # pyrefly: ignore [unexpected-keyword]
            num_stages=1,
            # pyrefly: ignore [unexpected-keyword]
            matrix_instr_nonkdim=16,
            # pyrefly: ignore [unexpected-keyword]
            waves_per_eu=1,
            # pyrefly: ignore [unexpected-keyword]
            regclass_priority_trumps_globalness=bwd_variant == "v3_ttgir_fp32_pipeline_qdo",
        )
        return (
            dq_final if dq_final is not None else dq_v3.to(q.dtype),
            dk_v3,
            dv_v3.reshape(-1),
        )

    # The generic kernel reduces dQ through FP32 memory.
    use_per_kv_head = G > 1
    dq = switch_to_contiguous_if_needed(torch.empty_like(q, dtype=torch.float32))
    dk = switch_to_contiguous_if_needed(torch.empty_like(k, dtype=k.dtype))
    dv = dk if shared_kv else switch_to_contiguous_if_needed(torch.empty_like(v, dtype=k.dtype))

    if use_per_kv_head:
        H_kv = H // G
        grid = lambda meta: (Z * H_kv, )  # noqa E731
    else:
        grid = lambda meta: (Z * H, 1)  # noqa E731
    _tlx_gfx950_cross_attn_bwd[grid](
        Q=q,
        K=k,
        V=v,
        DO=dout,
        seq_offsets=seq_offsets,
        seq_offsets_q=seq_offsets_q,
        DQ=dq,
        DK=dk,
        DV=dv,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_dom=dout.stride(0),
        stride_doh=dout.stride(1),
        stride_dqm=dq.stride(0),
        stride_dqh=dq.stride(1),
        stride_dkn=dk.stride(0),
        stride_dkh=dk.stride(1),
        stride_dvn=dv.stride(0),
        stride_dvh=dv.stride(1),
        alpha=alpha,
        max_seq_len=max_seq_len,
        attn_scale=attn_scale,
        M=M,
        Delta=Delta,
        stride_mm=stride_mm,
        num_targets=num_targets,
        Z=Z,
        AUTOTUNE_Z=next_power_of_2(Z),
        H=H,
        G=G,
        num_softmax_heads=num_softmax_heads,
        max_q_len=next_power_of_2(max_q_len),
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
        DimQ=DimQ,
        DimV=DimV,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        SHARED_KV=shared_kv,
        TRUNCATE_METHOD=truncate_method,
        HAS_CAUSAL=causal,
        HAS_NUM_TARGETS=num_targets is not None,
        PER_KV_HEAD=use_per_kv_head,
    )

    # The functional operator schema requires distinct output storage.
    # Shared KV uses one combined gradient and an empty dv placeholder.
    if shared_kv:
        dv = torch.empty(0, dtype=v.dtype, device=v.device)
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


def _tlx_gfx950_cross_attn_double_backward(ctx, *grads):
    raise RuntimeError("Trying to backward through tlx.hstu_cross_attn_gfx950_bwd.default but no autograd "
                       "formula was registered. Please use register_autograd to add one.")


# Extra dispatch keys require the standard backend dispatch path.
_HSTU_BWD_EAGER_KEYSETS = tuple(
    (torch._C.DispatchKeySet(backend) | torch._C.DispatchKeySet(autograd)).raw_repr()
    for backend, autograd in ((torch._C.DispatchKey.CPU, torch._C.DispatchKey.AutogradCPU),
                              (torch._C.DispatchKey.CUDA, torch._C.DispatchKey.AutogradCUDA)))


def _tlx_gfx950_cross_attn_bwd_dispatch(keyset, *args):
    unsafe = keyset.raw_repr() not in _HSTU_BWD_EAGER_KEYSETS
    # Forward AD dual tensors can have ordinary dispatch keys.
    unsafe = unsafe or getattr(torch.autograd.forward_ad, "_current_level", 0) >= 0
    unsafe = unsafe or any(isinstance(arg, torch.Tensor) and type(arg) is not torch.Tensor for arg in args)
    if unsafe:
        with torch._C._AutoDispatchBelowAutograd():
            return tlx_gfx950_cross_attn_bwd.redispatch(keyset & torch._C._after_autograd_keyset, *args)
    return _tlx_gfx950_cross_attn_bwd_impl(*args)


class _TlxGfx950CrossAttentionBackwardAutogradFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, keyset, *args):
        return _tlx_gfx950_cross_attn_bwd_dispatch(keyset, *args)

    @staticmethod
    def backward(ctx, *grads):
        return _tlx_gfx950_cross_attn_double_backward(ctx, *grads)


def _tlx_gfx950_cross_attn_bwd_autograd(keyset, *args):
    if torch.is_grad_enabled() and torch._C._any_requires_grad(*args):
        return _TlxGfx950CrossAttentionBackwardAutogradFunction.apply(keyset, *args)
    return _tlx_gfx950_cross_attn_bwd_dispatch(keyset, *args)


# Keep the library alive for the lifetime of its dispatcher registrations.
_hstu_cross_attn_bwd_library = torch.library.Library("tlx", "FRAGMENT")
_hstu_cross_attn_bwd_library.define(
    "hstu_cross_attn_gfx950_bwd" + torch.library.infer_schema(_tlx_gfx950_cross_attn_bwd_impl, mutates_args=()),
    tags=(torch.Tag.pt2_compliant_tag, ),
)
# The implementation allocates distinct outputs and does not mutate inputs.
# Direct registration avoids custom_op's runtime alias-check wrapper.
_hstu_cross_attn_bwd_library.impl("hstu_cross_attn_gfx950_bwd", _tlx_gfx950_cross_attn_bwd_impl,
                                  "CompositeExplicitAutograd")
_hstu_cross_attn_bwd_library.impl("hstu_cross_attn_gfx950_bwd", _tlx_gfx950_cross_attn_bwd_autograd, "Autograd",
                                  with_keyset=True)
tlx_gfx950_cross_attn_bwd = torch.ops.tlx.hstu_cross_attn_gfx950_bwd.default


@torch.library.register_fake("tlx::hstu_cross_attn_gfx950_bwd", lib=_hstu_cross_attn_bwd_library)
def _tlx_gfx950_cross_attn_bwd_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    max_seq_len: int,
    alpha: float,
    max_q_len: Optional[int],
    seq_offsets_q: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    causal: bool,
    shared_kv: bool,
    G: int,
    num_softmax_heads: int = 0,
    M: Optional[torch.Tensor] = None,
    Delta: Optional[torch.Tensor] = None,
    stride_mm: int = 0,
    truncate_method: str = "none",
    enable_tma: bool = False,
    bwd_variant: Optional[str] = None,
    OUT: Optional[torch.Tensor] = None,
    backward_route: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v) if not shared_kv else torch.empty(0, dtype=v.dtype, device=v.device),
    )


class _TlxGfx950CrossAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        max_seq_len: int,
        alpha: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: Optional[torch.Tensor],
        seq_offsets: torch.Tensor,
        attn_scale: torch.Tensor,
        seq_offsets_q: torch.Tensor,
        max_q_len: int,
        shared_kv: bool,
        num_softmax_heads: int,
        enable_tma: bool,
        truncate_method: str,
        num_targets: Optional[torch.Tensor] = None,
        causal: bool = False,
        bwd_variant: Optional[str] = None,
    ):
        _validate_backward_variant(bwd_variant, truncate_method)
        q = switch_to_contiguous_if_needed(q)
        k = switch_to_contiguous_if_needed(k)
        if not shared_kv:
            assert v is not None, "v is required when shared_kv is False"
            v = switch_to_contiguous_if_needed(v)
        else:
            v = k
        total_seq_len_q, H, DimQ = q.shape
        _, H_kv, DimV = v.shape
        assert H % H_kv == 0, f"H ({H}) must be divisible by H_kv ({H_kv})"
        G = H // H_kv  # GQA group size (number of Q heads per KV head)
        if total_seq_len_q == 0:
            out = torch.zeros(total_seq_len_q, H, DimV, device=q.device, dtype=q.dtype)
            return out

        if (G > 1 and H_kv == 1 and (num_softmax_heads == 0 or num_softmax_heads == H) and not causal):
            # GQA can not enabled with causal masking
            seq_offsets_q = seq_offsets_q * G
            max_q_len = max_q_len * G
            num_softmax_heads = num_softmax_heads // G
            q = q.view(total_seq_len_q * G, H_kv, DimQ)
            G = 1

        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        assert HEAD_DIM_V in {16, 32, 64, 128, 256}

        if bwd_variant == "split_softmax":
            torch._assert(
                truncate_method == "none",
                "Split-softmax does not support truncation",
            )
            torch._assert_async(
                torch.all(seq_offsets[1:] - seq_offsets[:-1] <= max_seq_len),
                "Split-softmax KV lengths must fit max_seq_len",
            )
            torch._assert_async(
                torch.all(seq_offsets_q[1:] - seq_offsets_q[:-1] <= max_q_len),
                "Split-softmax query lengths must fit max_q_len",
            )

        ctx.bwd_variant = bwd_variant
        out, M = tlx_gfx950_cross_attn_fwd(
            max_seq_len=max_seq_len,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            seq_offsets_q=seq_offsets_q,
            max_q_len=max_q_len,
            attn_scale=attn_scale,
            shared_kv=shared_kv,
            num_softmax_heads=num_softmax_heads,
            G=G,
            enable_tma=enable_tma,
            truncate_method=truncate_method,
            num_targets=num_targets,
            causal=causal,
        )

        saved_tensors = [q, k, v, seq_offsets, attn_scale, seq_offsets_q]
        if num_softmax_heads > 0:
            saved_tensors.extend([M, out])
        if num_targets is not None:
            saved_tensors.append(num_targets)
        ctx.has_num_targets = num_targets is not None
        ctx.causal = causal
        ctx.save_for_backward(*saved_tensors)
        ctx.alpha = alpha
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.max_seq_len = max_seq_len
        ctx.max_q_len = max_q_len
        ctx.truncate_method = truncate_method
        ctx.shared_kv = shared_kv
        ctx.enable_tma = enable_tma
        ctx.num_softmax_heads = num_softmax_heads
        ctx.total_seq_len_q = total_seq_len_q
        ctx.G = G
        ctx.H = H
        ctx.H_kv = H_kv
        batch = seq_offsets.numel() - 1
        ctx.use_small_backward = (bwd_variant in (None, "fp32_pipeline_qdo", "v3_ttgir_fp32_pipeline_qdo")
                                  and 0 < batch <= 64 and 0 < max_seq_len <= 2048 and max_q_len == 256 and H > 0
                                  and H == H_kv and k.shape[1] == H and num_softmax_heads == H and DimQ == 128
                                  and DimV == 128 and not causal and not enable_tma and truncate_method == "none"
                                  and q.dtype == k.dtype == v.dtype == torch.bfloat16 and q.is_contiguous()
                                  and k.is_contiguous() and v.is_contiguous() and attn_scale.ndim == 0)
        ctx.use_retained_backward = (
            bwd_variant in (None, "fp32_pipeline_qdo", "v3_ttgir_fp32_pipeline_qdo") and batch >= 256
            and max_seq_len in (256, 512, 1024, 2048) and max_q_len == 256 and shared_kv
            and H == H_kv == num_softmax_heads == 1 and DimQ == DimV == 128 and not causal and not enable_tma
            and truncate_method == "none" and q.dtype == k.dtype == torch.bfloat16 and q.is_contiguous()
            and k.is_contiguous() and q.device.type == "cuda" and q.device == k.device
            and seq_offsets.ndim == seq_offsets_q.ndim == 1 and seq_offsets.numel() == seq_offsets_q.numel()
            and seq_offsets.dtype == seq_offsets_q.dtype == torch.int64
            and seq_offsets.device == seq_offsets_q.device == q.device and seq_offsets.is_contiguous()
            and seq_offsets_q.is_contiguous() and attn_scale.ndim == 0 and torch.version.hip is not None
            and torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0] == "gfx950")
        # Limit default pipeline dispatch to the validated B2048 KV128/KV384 workloads.
        ctx.use_pipeline_backward = (
            bwd_variant is None and batch == 2048 and max_seq_len in (128, 384) and max_q_len == 256 and shared_kv
            and H == H_kv == num_softmax_heads == 1 and DimQ == DimV == 128 and not causal and not enable_tma
            and truncate_method == "none" and q.dtype == k.dtype == torch.bfloat16 and q.is_contiguous()
            and k.is_contiguous() and q.device.type == "cuda" and q.device == k.device
            and seq_offsets.ndim == seq_offsets_q.ndim == 1 and seq_offsets.numel() == seq_offsets_q.numel()
            and seq_offsets.dtype == seq_offsets_q.dtype == torch.int64
            and seq_offsets.device == seq_offsets_q.device == q.device and seq_offsets.is_contiguous()
            and seq_offsets_q.is_contiguous() and num_targets is not None and attn_scale.ndim == 0
            and torch.version.hip is not None
            and torch.cuda.get_device_properties(q.device).gcnArchName.split(":")[0] == "gfx950")
        if ctx.use_pipeline_backward:
            ctx.bwd_variant = "fp32_pipeline_qdo"
        ctx.use_coarse_backward = ctx.use_small_backward and batch > 4 and max_seq_len >= 384 and H == 1 and shared_kv
        ctx.cache_coarse_delta = ctx.use_coarse_backward and max_seq_len >= 1024
        ctx.use_compact_backward = (ctx.use_small_backward and batch <= 4 and (max_seq_len <= 1024 or shared_kv))
        return out.view(total_seq_len_q, H, DimV)

    @staticmethod
    # pyre-ignore[14]
    def backward(  # noqa C901
        ctx, dout: torch.Tensor) -> Tuple[
            None,
            None,
            torch.Tensor,
            torch.Tensor,
            Optional[torch.Tensor],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]:
        saved_tensors = ctx.saved_tensors
        q, k, v, seq_offsets, attn_scale, seq_offsets_q = saved_tensors[:6]
        # Keep JIT launchers inside the custom backward op.
        if ctx.use_retained_backward:
            backward_route = "retained"
        elif ctx.use_coarse_backward:
            backward_route = "coarse_cached" if ctx.cache_coarse_delta else "coarse"
        elif ctx.use_compact_backward:
            backward_route = "compact"
        elif ctx.use_small_backward:
            backward_route = "small"
        else:
            backward_route = None
        dout = dout.contiguous()
        idx = 6
        num_softmax_heads = ctx.num_softmax_heads
        # Reshape dout from [T, H, DimV] to folded [T*G, H_kv, DimV].
        #
        # This check is inherited from the source and does not fire for the
        # all-softmax fold: forward stores the post-fold `num_softmax_heads`
        # (H // G, so 1) while `ctx.H` is the pre-fold count, so the
        # `num_softmax_heads == ctx.H` arm is never true there. It is inert
        # either way. `dout` is contiguous, so viewing [T, H, DimV] as
        # [T*G, H_kv, DimV] moves no data, and the op below reshapes to
        # exactly that using the saved folded q's shape whether or not this
        # branch ran. Verified on H=4, H_kv=1, non-causal with
        # num_softmax_heads in {0, 4}: gradients match the reference either
        # way.
        if (ctx.H > 1 and ctx.H_kv == 1 and (num_softmax_heads == 0 or num_softmax_heads == ctx.H) and not ctx.causal):
            dout = dout.view(ctx.total_seq_len_q * ctx.H, -1, dout.shape[-1])
        if num_softmax_heads > 0:
            M = saved_tensors[idx].contiguous()
            idx += 1
            out = saved_tensors[idx].contiguous()
            idx += 1
            Delta = None
        else:
            M = torch.empty(0, device=q.device, dtype=torch.float32)
            Delta = None
            out = None
        if ctx.has_num_targets:
            num_targets = saved_tensors[idx].contiguous()
            idx += 1
        else:
            num_targets = None
        dq, dk, dv = torch.ops.tlx.hstu_cross_attn_gfx950_bwd.default(
            dout,
            q,
            k,
            v,
            seq_offsets,
            attn_scale,
            ctx.max_seq_len,
            ctx.alpha,
            ctx.max_q_len,
            seq_offsets_q,
            num_targets,
            ctx.causal,
            ctx.shared_kv,
            ctx.G,
            num_softmax_heads,
            M,
            Delta,
            0,  # stride_mm
            ctx.truncate_method,
            ctx.enable_tma,
            ctx.bwd_variant,
            out,
            backward_route,
        )
        dq = dq.view(ctx.total_seq_len_q, ctx.H, -1)
        return (
            None,
            None,
            dq,
            dk,
            dv if not ctx.shared_kv else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@torch.jit.unused
@torch.fx.wrap
def tlx_gfx950_cross_attn_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    max_q_len: Optional[int] = None,
    seq_offsets_q: Optional[torch.Tensor] = None,
    sort_by_length: bool = False,
    num_softmax_heads: int = 0,
    num_targets: Optional[torch.Tensor] = None,
    causal: bool = False,
    shared_kv: bool = False,
    enable_tma: bool = False,
    actual_max_seq_len: int = 0,
    truncate_method: str = "none",
    bwd_variant: Optional[str] = None,
) -> torch.Tensor:
    # Self-attention-style usage: q and kv are the same sequence. The Triton source
    # resolves this in its backward op, so it has to be resolved here instead.
    if max_q_len is None:
        max_q_len = max_seq_len
        assert seq_offsets_q is None
        seq_offsets_q = seq_offsets

    # Dynamo cannot trace autograd.Function.apply when the same tensor object
    # is passed to multiple forward inputs. Preserve aliasing with a zero-copy
    # view while giving k and v distinct Python objects.
    if v is k:
        v = v.view(v.shape)
    return _TlxGfx950CrossAttentionFunction.apply(
        max_seq_len,
        alpha,
        q,
        k,
        v if not shared_kv else None,
        seq_offsets,
        attn_scale,
        seq_offsets_q,
        max_q_len,
        shared_kv,
        num_softmax_heads,
        enable_tma,
        truncate_method,
        num_targets,
        causal,
        bwd_variant,
    )


def tlx_gfx950_cross_attn_mha_wrapper(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_scale: torch.Tensor,
    max_q_len: Optional[int] = None,
    seq_offsets_q: Optional[torch.Tensor] = None,
    sort_by_length: bool = False,
    num_softmax_heads: int = 0,
    num_targets: Optional[torch.Tensor] = None,
    causal: bool = False,
    shared_kv: bool = False,
    enable_tma: bool = False,
    actual_max_seq_len: int = 0,
    truncate_method: str = "none",
    bwd_variant: Optional[str] = None,
) -> torch.Tensor:
    assert not sort_by_length, ("sort_by_length not supported by the cross attention kernel")
    return tlx_gfx950_cross_attn_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        attn_scale=attn_scale,
        max_q_len=max_q_len,
        seq_offsets_q=seq_offsets_q,
        sort_by_length=sort_by_length,
        num_softmax_heads=num_softmax_heads,
        num_targets=num_targets,
        causal=causal,
        shared_kv=shared_kv,
        enable_tma=enable_tma,
        actual_max_seq_len=actual_max_seq_len,
        truncate_method=truncate_method,
        bwd_variant=bwd_variant,
    )


@tlx_gfx950_cross_attn_fwd.register_fake
@tlx_gfx950_cross_attn_fwd.register_kernel("cpu")
def _tlx_gfx950_cross_attn_fwd_fake(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_offsets_q: torch.Tensor,
    max_q_len: int,
    attn_scale: torch.Tensor,
    G: int,
    shared_kv: bool = False,
    num_softmax_heads: int = 0,
    enable_tma: bool = False,
    truncate_method: str = "none",
    num_targets: Optional[torch.Tensor] = None,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    L, H, _ = q.shape
    if num_softmax_heads > 0:
        M = torch.empty((L, num_softmax_heads), dtype=torch.float32, device=v.device)
    else:
        # The real op returns a 0-element M here. The Triton source's fake returns
        # `torch.empty(1)`, which reports the wrong size to `torch.compile` /
        # `torch.export` and fails `torch.library.opcheck`.
        M = torch.empty(0, dtype=torch.float32, device=v.device)
    return (
        torch.empty(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype),
        M,
    )


def hstu_cross_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_offsets_q: torch.Tensor,
    max_seq_len: int,
    max_q_len: int,
    attn_scale: torch.Tensor,
    alpha: float,
    num_softmax_heads: int = 0,
    num_targets: Optional[torch.Tensor] = None,
    causal: bool = False,
    shared_kv: bool = False,
    *,
    space: str = "curated",
) -> torch.Tensor:
    if space not in ("curated", "full", "smoke"):
        raise ValueError(f"hstu_cross_attn does not provide space={space!r}")
    global _SPACE
    _SPACE = space
    return tlx_gfx950_cross_attn_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        attn_scale=attn_scale,
        max_q_len=max_q_len,
        seq_offsets_q=seq_offsets_q,
        num_softmax_heads=num_softmax_heads,
        num_targets=num_targets,
        causal=causal,
        shared_kv=shared_kv,
    )
