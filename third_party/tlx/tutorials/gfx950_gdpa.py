"""GDPA (Generalized Dot-Product Attention) forward for AMD gfx950 / MI350 (CDNA4).

GDPA is flash attention with the online softmax replaced by a pointwise
activation:

    qk  = q @ k.T * qk_scale
    p   = gelu(qk)              # no row-max, no running sum, no acc rescale
    out = p @ v

Because there is no softmax there are no cross-lane reductions and no
accumulator correction between the two dots, so the inner loop is strictly
simpler than FA's.

v1 scope (the OmniFM V3 pFFN production path):
  * jagged Q   -- variable sequence length per batch, `q_offsets[B + 1]`
  * dense KV   -- every batch has exactly `dff` key/value rows
  * fast gelu, non-causal, no bias, no window, no GQA, no fused QKV
Everything else in the Blackwell kernel's constexpr surface (FUSED_QKV,
BROADCAST_Q, WINDOW_SIZE, STAGE, is_predict, ...) is out of scope here.

Tensor layouts (all bf16, `d = D // H`):
    q    [total_q, H, d]     jagged, row i of batch b at q_offsets[b] + i
    k    [B * dff, H, d]     dense, batch b occupies rows b*dff .. (b+1)*dff
    v    [B * dff, H, d]     dense, same as k
    out  [total_q, H, d]     jagged, same layout as q

These kernels are gfx950/MI350 (CDNA4)-specific.
"""

import math
from typing import Any, Optional

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

try:
    from triton.language.extra.libdevice import (
        fast_dividef,
        fast_expf,
    )  # @manual=//triton:triton
except ImportError:
    try:
        # @manual=//triton:triton
        from triton.language.extra.hip.libdevice import fast_dividef, fast_expf
    except ImportError:
        # pyre-ignore[21]
        from triton.language.math import (
            fast_dividef,
            fast_expf,
        )  # @manual=//triton:triton

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# The production shape this kernel targets: OmniFM V3 pFFN.
#   batch=2048 max_seq_len=500 dim=256 head=4 head_dim=64 kv_len(dff)=256 sp=0.68
# Reference points on this shape (bf16, forward), MI350X, stock Triton GDPA:
#   tuned (all AMD opts)  0.5927 ms / 144.9 TFLOPS   <- the number to beat
#   baseline              1.4714 ms /  58.4 TFLOPS
# Those numbers correspond to total_q = 327,680 (= B * 160), which the
# "uniform" seq-len mode reproduces exactly; see SEQ_LEN_MODES.
PROD_CONFIG: dict[str, Any] = {
    "B": 2048,
    "max_M": 500,
    "D": 256,
    "H": 4,
    "dff": 256,
    "sparsity": 0.68,
    "dtype": torch.bfloat16,
    "seq_len_mode": "uniform",
}

# ═══════════════════════════════════════════════════════════════════════════
# Activation
# ═══════════════════════════════════════════════════════════════════════════
#
# The kernel reproduces the production `fast_gelu` (activation_enum_int == 3)
# from fbcode/ads_mkl/ops/triton/math.py, AMD branch:
#
#     k = 2 * 0.7978845608
#     fast_gelu(x) = x * sigmoid_approx(k * x * (1 + 0.044715 * x^2))
#     sigmoid_approx(u) = fast_dividef(1, 1 + fast_expf(-u))
#
# This is the *tanh* approximation of gelu, rewritten through the identity
#   0.5 * (1 + tanh(u)) == sigmoid(2u)
# so it computes with HIP's fast_dividef/fast_expf instead of a tanh. It is
# therefore numerically equivalent to torch.nn.GELU(approximate="tanh") --
# which is exactly what ads_mkl's get_pytorch_activation maps FastGeLU to --
# and NOT the cruder x * sigmoid(1.702x) form.
#
# The sigmoid rewrite is what makes this cheap on gfx950: the Blackwell path
# uses `tanh_approx_fp32`, which is CUDA inline PTX (`tanh.approx.f32`) with no
# gfx950 equivalent. This substitution is the +47.7% step in the MI350X GDPA
# tuning ladder.
#
# The correctness reference below uses *exact* erf gelu, so the accuracy gate
# absorbs the approximation error on purpose. `gelu_approx_error()` reports the
# approximation's own contribution separately, so a failing test can be
# attributed to either the kernel or the approximation rather than guessed at.

# k = 2 * sqrt(2/pi), matching ads_mkl.
#
# Two spellings on purpose: `_fast_gelu` is @triton.jit and reads these as
# module globals, which the code generator only permits for constexpr values
# (plain floats raise NameError at compile time), while the torch mirror below
# needs real Python floats to combine with tensors.
_GELU_TANH_K = 2.0 * 0.7978845608
_GELU_TANH_C = 0.044715
GELU_TANH_K = tl.constexpr(_GELU_TANH_K)
GELU_TANH_C = tl.constexpr(_GELU_TANH_C)


def gelu_exact(x: torch.Tensor) -> torch.Tensor:
    """Exact gelu: x * 0.5 * (1 + erf(x / sqrt(2))). Computed in fp32."""
    x = x.float()
    return x * 0.5 * (1.0 + torch.erf(x * (1.0 / math.sqrt(2.0))))


def fast_gelu_ref(x: torch.Tensor) -> torch.Tensor:
    """Torch mirror of the kernel's fast_gelu -- the ads_mkl AMD formula."""
    x = x.float()
    return x * torch.sigmoid(_GELU_TANH_K * x * (1.0 + _GELU_TANH_C * x * x))


# ═══════════════════════════════════════════════════════════════════════════
# Data generation
# ═══════════════════════════════════════════════════════════════════════════

# Two incompatible readings of "sparsity" are in play, and they differ by 2.1x
# in total work on the production shape. Both are supported explicitly rather
# than silently picking one:
#
#   "uniform" -- every sequence is exactly round((1 - sparsity) * max_M).
#       At B=2048, max_M=500, sparsity=0.68 this gives seq=160 and
#       total_q = 2048 * 160 = 327,680, which reproduces the total_q behind the
#       144.9 TFLOPS MI350X reference number exactly. Use this when comparing
#       against that baseline.
#
#   "random"  -- the Blackwell GDPA generator: randint over
#       [(2*sparsity - 1) * max_M, max_M). At the same settings this gives an
#       average sequence of ~337 and total_q ~690k. Genuinely jagged, so it is
#       the better stress test for the tile scheduler, but its latency is not
#       comparable to the reference number.
SEQ_LEN_MODES = ("uniform", "random")


def generate_sparse_seq_len(
    B: int,
    max_seq_len: int,
    sparsity: float,
    device: torch.device | str,
    mode: str = "uniform",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Per-batch Q lengths. See SEQ_LEN_MODES for the two conventions."""
    assert mode in SEQ_LEN_MODES, f"mode must be one of {SEQ_LEN_MODES}, got {mode!r}"

    if sparsity == 1.0:
        return torch.full((B, ), max_seq_len, device=device, dtype=torch.int32)

    if mode == "uniform":
        seq = max(int(round((1.0 - sparsity) * max_seq_len)), 1)
        return torch.full((B, ), seq, device=device, dtype=torch.int32)

    if sparsity >= 0.5:
        low = max(int((2 * sparsity - 1.0) * max_seq_len), 1)
        high = max_seq_len
    else:
        low = 1
        high = max(int(2 * sparsity * max_seq_len), 2)
    return torch.randint(low=low, high=high, size=(B, ), device=device, dtype=torch.int32, generator=generator)


def generate_gdpa_data(
    B: int,
    max_M: int,
    D: int,
    H: int,
    dff: int,
    sparsity: float = 0.68,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str | None = None,
    seed: int = 42,
    seq_len_mode: str = "uniform",
) -> dict[str, Any]:
    """Jagged-Q / dense-KV GDPA inputs.

    Returns q/k/v plus `q_offsets` and the derived sizes. `D` is the *model*
    dimension; per-head width is `d = D // H`.

    `seq_len_mode` picks the sparsity convention -- see SEQ_LEN_MODES. It
    changes total work by ~2x on the production shape, so it must be reported
    alongside any measurement.

    Values are drawn from a unit normal so the qk products land in the range
    where the gelu approximation is least accurate -- this is deliberately the
    hard case for the accuracy gate.
    """
    if device is None:
        device = DEVICE
    assert D % H == 0, f"D={D} must be divisible by H={H}"
    d = D // H

    gen = torch.Generator(device=device).manual_seed(seed)

    seq_lens = generate_sparse_seq_len(B, max_M, sparsity, device, mode=seq_len_mode, generator=gen)
    q_offsets = torch.cat(
        [torch.zeros(1, device=device, dtype=torch.int32),
         seq_lens.cumsum(dim=0).to(torch.int32)],
        dim=0,
    )
    total_q = int(q_offsets[-1].item())

    q = torch.randn(total_q, H, d, device=device, dtype=dtype, generator=gen)
    k = torch.randn(B * dff, H, d, device=device, dtype=dtype, generator=gen)
    v = torch.randn(B * dff, H, d, device=device, dtype=dtype, generator=gen)

    return {
        "q": q.contiguous(),
        "k": k.contiguous(),
        "v": v.contiguous(),
        "q_offsets": q_offsets,
        "seq_lens": seq_lens,
        "B": B,
        "H": H,
        "D": D,
        "d": d,
        "dff": dff,
        "max_M": max_M,
        "total_q": total_q,
        "sparsity": sparsity,
        "seq_len_mode": seq_len_mode,
        "dtype": dtype,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Reference
# ═══════════════════════════════════════════════════════════════════════════


def gdpa_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    dff: int,
    qk_scale: float = 1.0,
    activation=gelu_exact,
) -> torch.Tensor:
    """Torch reference: per (batch, head), out = activation(q @ k.T * s) @ v.

    Accumulates in fp32 and casts back to q's dtype at the end, matching what
    the kernel does (fp32 MFMA accumulators, bf16 store).

    Deliberately a plain loop over batches -- this is a correctness oracle, not
    a performance baseline, and the jagged Q makes a batched form awkward
    without padding (which would change the numerics).
    """
    B = q_offsets.numel() - 1
    out = torch.empty_like(q)
    offsets = q_offsets.tolist()

    for b in range(B):
        lo, hi = offsets[b], offsets[b + 1]
        if hi == lo:
            continue
        # [Mb, H, d] -> [H, Mb, d]
        q_b = q[lo:hi].transpose(0, 1).float()
        k_b = k[b * dff:(b + 1) * dff].transpose(0, 1).float()
        v_b = v[b * dff:(b + 1) * dff].transpose(0, 1).float()

        s = torch.bmm(q_b, k_b.transpose(1, 2)) * qk_scale  # [H, Mb, dff]
        p = activation(s)
        o = torch.bmm(p, v_b)  # [H, Mb, d]
        out[lo:hi] = o.transpose(0, 1).to(out.dtype)

    return out


def gelu_approx_error(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    dff: int,
    qk_scale: float = 1.0,
) -> dict[str, float]:
    """How much of the kernel-vs-reference gap is the gelu approximation alone.

    Runs the reference twice -- once with exact gelu, once with the sigmoid
    form the kernel uses -- and reports the delta. Any kernel error below this
    floor is not distinguishable from the approximation.
    """
    ref_exact = gdpa_ref(q, k, v, q_offsets, dff, qk_scale, activation=gelu_exact).float()
    ref_fast = gdpa_ref(q, k, v, q_offsets, dff, qk_scale, activation=fast_gelu_ref).float()
    diff = (ref_exact - ref_fast).abs()
    denom = ref_exact.abs().clamp_min(1e-6)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": (diff / denom).max().item(),
        "rel_l2": (diff.norm() / ref_exact.norm().clamp_min(1e-6)).item(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# TFLOPS
# ═══════════════════════════════════════════════════════════════════════════


def gdpa_flops(total_q: int, H: int, d: int, dff: int) -> int:
    """Forward FLOPs: two GEMMs per (row, head).

    qk  : [total_q, d] @ [d, dff]  -> 2 * total_q * d * dff
    pv  : [total_q, dff] @ [dff, d] -> 2 * total_q * dff * d
    Times H heads. Matches the formula used in the MI350X GDPA tuning report
    (`fwd_flops = 4 * H * d * total_Q * kv_len`), so numbers are directly
    comparable to the 144.9 TFLOPS baseline.
    """
    return 4 * H * d * total_q * dff


def gdpa_tflops(ms: float, total_q: int, H: int, d: int, dff: int) -> float:
    return gdpa_flops(total_q, H, d, dff) * 1e-12 / (ms * 1e-3)


# ═══════════════════════════════════════════════════════════════════════════
# Kernels
# ═══════════════════════════════════════════════════════════════════════════


@triton.jit
def _fast_gelu(x):
    """ads_mkl's AMD `fast_gelu`, verbatim: x * sigmoid(k*x*(1 + c*x^2)).

    Written as the sigmoid of the tanh argument (see the module header) so it
    lowers to fast_expf/fast_dividef; gfx950 has no tanh.approx.f32.
    """
    u = x * (GELU_TANH_K + GELU_TANH_K * GELU_TANH_C * x * x)
    return x * fast_dividef(1.0, 1.0 + fast_expf(-u))


@triton.jit
def _remap_xcd(pid, grid_size, NUM_XCDS: tl.constexpr):
    """Round-robin pid -> XCD so consecutive tiles of one (batch, head) land on
    the same chiplet and share its L2 slice for K/V."""
    pids_per_xcd = (grid_size + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = grid_size % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid
    return pid


@triton.jit
def _gdpa_pid_decode(H, MAX_M, BLOCK_M: tl.constexpr, NUM_XCDS: tl.constexpr):
    """Flat pid -> (batch, head, m-tile). Grid is sized on MAX_M, so tiles past a
    sequence's end are launched and exit immediately."""
    num_m_blocks = tl.cdiv(MAX_M, BLOCK_M)
    tpid = _remap_xcd(tl.program_id(0), tl.num_programs(0), NUM_XCDS)
    off_h = tpid % H
    off_nz = tpid // H
    pid_m = off_nz % num_m_blocks
    off_z = off_nz // num_m_blocks
    return off_z, off_h, pid_m


@triton.jit
def _gdpa_assume_strides(stride_qm, stride_qh, stride_kn, stride_kh, stride_vn, stride_vh, stride_om, stride_oh):
    tl.assume(stride_qm > 0)
    tl.assume(stride_qh > 0)
    tl.assume(stride_kn > 0)
    tl.assume(stride_kh > 0)
    tl.assume(stride_vn > 0)
    tl.assume(stride_vh > 0)
    tl.assume(stride_om > 0)
    tl.assume(stride_oh > 0)


@triton.jit
def _gdpa_fwd_pipelined(
    Q,
    K,
    V,
    Out,
    q_offsets,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    H,
    MAX_M,
    QK_SCALE: tl.constexpr,
    DFF: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
):
    """Modulo-scheduled GDPA forward.

    Prologue loads KV block 0; the steady-state loop consumes block i out of one
    LDS slot while `buffer_load_to_lds` fills block i+1 into the other; the last
    block is peeled so the hot loop never carries a mask. K is transposed at the
    memdesc level (`local_trans`), which is metadata-only -- it lands `local_load`
    directly in dot-operand layout and skips the ds_write/barrier/ds_read shuffle
    that `tl.dot(q, k.T)` would emit every iteration.

    Unlike flash attention there is no online softmax: `_fast_gelu` is pointwise,
    so no row reduction and no accumulator rescale sit between the two dots.
    """
    _gdpa_assume_strides(stride_qm, stride_qh, stride_kn, stride_kh, stride_vn, stride_vh, stride_om, stride_oh)
    off_z, off_h, pid_m = _gdpa_pid_decode(H, MAX_M, BLOCK_M, NUM_XCDS)

    seq_start = tl.load(q_offsets + off_z).to(tl.int64)
    seq_len = (tl.load(q_offsets + off_z + 1).to(tl.int64) - seq_start).to(tl.int32)
    start_m = pid_m * BLOCK_M

    if start_m < seq_len:
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        offs_dv = tl.arange(0, BLOCK_D_V)
        mask_m = offs_m < seq_len

        q_base = Q + off_h.to(tl.int64) * stride_qh + seq_start * stride_qm
        q = tl.load(q_base + offs_m[:, None] * stride_qm + offs_d[None, :], mask=mask_m[:, None], other=0.0)

        kv_start = off_z.to(tl.int64) * DFF
        k_ptrs = (K + off_h.to(tl.int64) * stride_kh + kv_start * stride_kn + offs_n[:, None] * stride_kn +
                  offs_d[None, :])
        v_ptrs = (V + off_h.to(tl.int64) * stride_vh + kv_start * stride_vn + offs_n[:, None] * stride_vn +
                  offs_dv[None, :])

        k_buf = tlx.local_alloc((BLOCK_N, BLOCK_D), K.dtype.element_ty, NUM_BUFFERS)
        v_buf = tlx.local_alloc((BLOCK_N, BLOCK_D_V), V.dtype.element_ty, NUM_BUFFERS)

        acc = tl.zeros([BLOCK_M, BLOCK_D_V], dtype=tl.float32)

        # KV is dense -- every batch has exactly DFF rows -- so the block count
        # is a compile-time constant and only the final block can be ragged.
        N_BLOCKS: tl.constexpr = (DFF + BLOCK_N - 1) // BLOCK_N
        N_MAIN: tl.constexpr = N_BLOCKS - 1
        EVEN_N: tl.constexpr = DFF % BLOCK_N == 0

        # Prologue: fetch block 0.
        if EVEN_N:
            tok_k0 = tlx.async_load(k_ptrs, tlx.local_view(k_buf, 0))
            tok_v0 = tlx.async_load(v_ptrs, tlx.local_view(v_buf, 0))
        else:
            mask0 = offs_n[:, None] < DFF
            tok_k0 = tlx.async_load(k_ptrs, tlx.local_view(k_buf, 0), mask=mask0)
            tok_v0 = tlx.async_load(v_ptrs, tlx.local_view(v_buf, 0), mask=mask0)
        tlx.async_load_commit_group([tok_k0, tok_v0])

        # Steady state: consume block i, prefetch block i+1 into the other slot.
        for block_id in tl.range(0, N_MAIN * BLOCK_N, BLOCK_N, num_stages=1):
            i = block_id // BLOCK_N
            slot_cur = i % NUM_BUFFERS
            slot_nxt = (i + 1) % NUM_BUFFERS
            next_off = block_id + BLOCK_N

            wait_tok = tlx.async_load_wait_group(0)
            kt_cur = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, slot_cur)), token=wait_tok)
            v_cur = tlx.local_load(tlx.local_view(v_buf, slot_cur), token=wait_tok)

            if EVEN_N:
                tok_k = tlx.async_load(k_ptrs + next_off * stride_kn, tlx.local_view(k_buf, slot_nxt))
                tok_v = tlx.async_load(v_ptrs + next_off * stride_vn, tlx.local_view(v_buf, slot_nxt))
            else:
                next_mask = (next_off + offs_n[:, None]) < DFF
                tok_k = tlx.async_load(k_ptrs + next_off * stride_kn, tlx.local_view(k_buf, slot_nxt), mask=next_mask)
                tok_v = tlx.async_load(v_ptrs + next_off * stride_vn, tlx.local_view(v_buf, slot_nxt), mask=next_mask)
            tlx.async_load_commit_group([tok_k, tok_v])

            qk = tl.dot(q, kt_cur, allow_tf32=True) * QK_SCALE
            p = _fast_gelu(qk)
            acc = tl.dot(p.to(v_cur.dtype), v_cur, acc, allow_tf32=True)

        # Peeled last block -- the only one that can need a boundary mask.
        SLOT_LAST: tl.constexpr = N_MAIN % NUM_BUFFERS
        wait_tok = tlx.async_load_wait_group(0)
        kt_cur = tlx.local_load(tlx.local_trans(tlx.local_view(k_buf, SLOT_LAST)), token=wait_tok)
        v_cur = tlx.local_load(tlx.local_view(v_buf, SLOT_LAST), token=wait_tok)

        qk = tl.dot(q, kt_cur, allow_tf32=True) * QK_SCALE
        p = _fast_gelu(qk)
        if not EVEN_N:
            kn_last = N_MAIN * BLOCK_N + offs_n
            p = tl.where(kn_last[None, :] < DFF, p, 0.0)
        acc = tl.dot(p.to(v_cur.dtype), v_cur, acc, allow_tf32=True)

        o_base = Out + off_h.to(tl.int64) * stride_oh + seq_start * stride_om
        tl.store(o_base + offs_m[:, None] * stride_om + offs_dv[None, :], acc.to(Out.dtype.element_ty),
                 mask=mask_m[:, None])


# ═══════════════════════════════════════════════════════════════════════════
# Launcher
# ═══════════════════════════════════════════════════════════════════════════

# `matrix_instr_nonkdim` and `waves_per_eu` are the two knobs that actually paid
# off in the MI350X stock-Triton GDPA tuning ladder (+29.8% for waves_per_eu on
# top of everything else), so they stay in the search space rather than pinned.
FULL_TUNING = False

# (BLOCK_M, BLOCK_N, num_warps, NUM_BUFFERS). BLOCK_M dominates: a longer m-tile
# amortises the K/V stream over more query rows, at the cost of more wasted rows
# in each sequence's ragged tail. NUM_BUFFERS is cheap to deepen here -- CDNA4
# has 160 KB of LDS and the widest entry below only needs 2 x (16 + 16) KB.
_FWD_TILES = (
    (128, 64, 4, 2),
    (128, 64, 4, 3),
    (256, 64, 8, 2),
    (256, 64, 8, 3),
    (256, 128, 8, 2),
)
_FWD_WAVES_PER_EU = (0, 2, 4)


def _get_fwd_configs() -> list[triton.Config]:
    if FULL_TUNING:
        tiles = [(bm, bn, nw, nb) for bm in (64, 128, 256) for bn in (32, 64, 128) for nw in (4, 8) for nb in (2, 3, 4)]
        waves = (0, 1, 2, 4)
        nonkdims = (16, 32)
    else:
        tiles = list(_FWD_TILES)
        waves = _FWD_WAVES_PER_EU
        nonkdims = (16, )

    return [
        triton.Config(
            {
                "BLOCK_M": bm,
                "BLOCK_N": bn,
                "NUM_BUFFERS": nb,
                "matrix_instr_nonkdim": nk,
                "waves_per_eu": we,
            },
            num_stages=1,
            num_warps=nw,
        ) for bm, bn, nw, nb in tiles for we in waves for nk in nonkdims
    ]


_AUTOTUNE_KEY = ["H", "MAX_M", "DFF", "QK_SCALE"]

_gdpa_fwd_pipelined_tuned = triton.autotune(configs=_get_fwd_configs(), key=_AUTOTUNE_KEY)(_gdpa_fwd_pipelined)


def _check_and_prepare(q, k, v, q_offsets, dff):
    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3, "q/k/v must be [rows, H, d]"
    total_q, H, d = q.shape
    B = q_offsets.numel() - 1
    assert k.shape == (B * dff, H, d), f"k must be [B*dff, H, d] = {(B * dff, H, d)}, got {tuple(k.shape)}"
    assert v.shape == k.shape, "v must have k's shape"
    assert d & (d - 1) == 0, f"head_dim must be a power of two, got {d}"
    assert q.stride(2) == 1 and k.stride(2) == 1 and v.stride(2) == 1, "q/k/v must be contiguous in d"
    return total_q, H, d, B


def _launch(kernel, q, k, v, q_offsets, dff, qk_scale, max_M, num_xcds, kw, config=None):
    total_q, H, d, B = _check_and_prepare(q, k, v, q_offsets, dff)
    out = torch.empty_like(q)
    if total_q == 0:
        return out

    if max_M is None:
        # One host sync. Pass `max_M` explicitly on any timed path.
        max_M = int((q_offsets[1:] - q_offsets[:-1]).max().item())
    if max_M == 0:
        return out

    if config is not None:
        # Pinned config: launch the JIT function directly, skipping the sweep.
        grid = (triton.cdiv(max_M, config["BLOCK_M"]) * B * H, )
        kernel = kernel.fn
        kw = {**kw, **config}
    else:
        grid = lambda meta: (triton.cdiv(max_M, meta["BLOCK_M"]) * B * H, )  # noqa: E731
    kernel[grid](
        q,
        k,
        v,
        out,
        q_offsets,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        out.stride(0),
        out.stride(1),
        H,
        max_M,
        QK_SCALE=qk_scale,
        DFF=dff,
        BLOCK_D=d,
        BLOCK_D_V=d,
        NUM_XCDS=num_xcds,
        **kw,
    )
    return out


def gdpa_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    dff: int,
    qk_scale: float = 1.0,
    max_M: Optional[int] = None,
    num_xcds: int = 8,
    config=None,
    **kw,
) -> torch.Tensor:
    """GDPA forward, jagged Q x dense KV.

    Args:
        q:         [total_q, H, d] bf16, jagged rows
        k, v:      [B * dff, H, d] bf16, dense per batch
        q_offsets: [B + 1] int32, prefix sum of per-batch Q lengths
        dff:       KV rows per batch
        qk_scale:  scale applied to q @ k.T before the activation
        max_M:     longest sequence, used to size the grid. Derived from
                   `q_offsets` when omitted, which costs a host sync -- pass it
                   explicitly from any benchmarked path.
        config:    Pin a kernel config and bypass the autotuner
                   (Blackwell/Hopper tutorial convention). None autotunes.

    Returns:
        [total_q, H, d], same dtype/layout as q.
    """
    return _launch(_gdpa_fwd_pipelined_tuned, q, k, v, q_offsets, dff, qk_scale, max_M, num_xcds, kw, config=config)


KERNEL_REGISTRY = {
    "pipelined": gdpa_forward,
}


def get_kernel(name):
    if name not in KERNEL_REGISTRY:
        raise ValueError(f"Unknown kernel: {name!r}. Available: {list(KERNEL_REGISTRY.keys())}")
    return KERNEL_REGISTRY[name]


# Default launcher for the suite.
gdpa = gdpa_forward
