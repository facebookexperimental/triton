# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Standalone repro of an AMD (gfx950) grouped-GEMM epilogue store corruption.

Specialized grouped GEMM (Y = X @ W^T), reduced to the minimal reproducing
form: manual LDS double-buffered pipeline (tlx.async_load -> buffer_load offen
lds), B staged transposed [BK,BN] in a K-contiguous swizzled_shared, A M-half
(top/bottom) split with separate pools, overlapped two-dot-pair epilogue. Inner
K-loop removed (butchered); nhuge (inf/nan/>1e30) counts the corruption.

The racing store is a flat 64-bit global_store_dwordx4 to a >2GB y with a
power-of-2 row stride; that combination is what corrupts.

    x [M, K] (row-packed), w [1, N, K], y [M, N].
"""
import collections
import glob
import os
import pathlib
import re
import shutil
import subprocess
import sys

import torch
import triton
import triton.language as tl
import triton.language.core as _tlc

# @manual=//triton:triton
import triton.language.extra.tlx as tlx
from triton.runtime.jit import constexpr_function


@_tlc.builtin
def _workgroup_barrier(_semantic=None):
    _semantic.builder.create_workgroup_barrier()


@constexpr_function
def _swizzled_b_layout(order):
    # Swizzled (non-padded) shared encoding for a K-contiguous bf16 MFMA operand:
    # 128-bit vectors (8 bf16), swizzle phases sized for the 16x16x32 tile.
    return tlx.swizzled_shared_layout_encoding(
        8, 1, 8, list(order), [1, 1], [1, 1], [1, 1], list(order)
    )


def _num_cus() -> int:
    return torch.cuda.get_device_properties("cuda").multi_processor_count


@triton.jit
def _amd_grouped_gemm_fprop_kernel(
    x_ptr,  # [M, K]
    w_ptr,  # [1, N, K]
    y_ptr,  # [M, N]
    M,  # runtime row count
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym: tl.constexpr,
    stride_yn: tl.constexpr,
    NUM_CUS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    tl.assume(stride_xm > 0)
    tl.assume(stride_wn > 0)
    tl.assume(stride_ym > 0)
    # x/w are row-contiguous: bounds for the buffer-op offset range analysis.
    tl.assume(stride_xk == 1)
    tl.assume(stride_xm == K)
    tl.assume(stride_wk == 1)
    tl.assume(stride_wn == K)

    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    HALF_M: tl.constexpr = BLOCK_SIZE_M // 2
    # A uses two [HALF_M,BK] pools (top/bottom).
    buffers_A_top = tlx.local_alloc(
        (HALF_M, BLOCK_SIZE_K), tlx.dtype_of(x_ptr), NUM_STAGES
    )
    buffers_A_bot = tlx.local_alloc(
        (HALF_M, BLOCK_SIZE_K), tlx.dtype_of(x_ptr), NUM_STAGES
    )
    # B staged transposed [BK,BN] (transpose in b_off indices).  The swizzled
    # shared hint is load-bearing at compile time -- a plain layout hits an
    # unrealized_conversion_cast -- though the buffer still lowers to padded.
    buffers_B = tlx.local_alloc(
        (BLOCK_SIZE_K, BLOCK_SIZE_N), tlx.dtype_of(w_ptr), NUM_STAGES,
        layout=_swizzled_b_layout([0, 1]),
    )

    tile_idx = pid
    num_m_tiles = tl.cdiv(M, BLOCK_SIZE_M)
    num_tiles = num_m_tiles * num_n_tiles

    # Claim tiles that fall on this program's stride.  The redundant
    # `tile_idx >= 0` guard is a tautology (pid>=0, stride>0) but its presence
    # in the loop header measurably widens the epilogue-store race window.
    while tile_idx >= 0 and tile_idx < num_tiles:
        tile_m = tile_idx // num_n_tiles
        tile_n = tile_idx % num_n_tiles

        offs_n = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_n_col = offs_n[None, :] < N

        # Buffer-op-friendly addressing: uniform int64 base + small invariant
        # offset (x may exceed 2GB -> no tt.pointer_range=32).
        m_block = (tile_m * BLOCK_SIZE_M).to(tl.int64)
        x_base = x_ptr + m_block * stride_xm
        a_off = (
            tl.arange(0, HALF_M)[:, None] * stride_xm + offs_k[None, :] * stride_xk
        )
        n_block = (tile_n * BLOCK_SIZE_N).to(tl.int64)
        wb = w_ptr + n_block * stride_wn
        b_off = (
            offs_k[:, None] * stride_wk
            + tl.arange(0, BLOCK_SIZE_N)[None, :] * stride_wn
        )

        # A split into two M-halves (top/bottom); B whole.
        offs_mt = tile_m * BLOCK_SIZE_M + tl.arange(0, HALF_M)
        offs_mb = offs_mt + HALF_M
        lmask_mt = offs_mt[:, None] < M
        lmask_mb = offs_mb[:, None] < M
        smask_mt = offs_mt[:, None] < M
        smask_mb = offs_mb[:, None] < M
        row_t = offs_mt.to(tl.int64)
        row_b = offs_mb.to(tl.int64)
        acc_top = tl.zeros((HALF_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_bot = tl.zeros((HALF_M, BLOCK_SIZE_N), dtype=tl.float32)

        kfull = offs_k < K
        a_mask_t = lmask_mt & kfull[None, :]
        a_mask_b = lmask_mb & kfull[None, :]
        b_mask = kfull[:, None] & mask_n_col

        # Prologue: issue async loads for both operands (B before A).
        for b in tl.static_range(2):
            tlx.async_load(
                wb + b * BLOCK_SIZE_K * stride_wk + b_off,
                tlx.local_view(buffers_B, b), mask=b_mask, other=0.0,
            )
            tlx.async_load_commit_group()
            tlx.async_load(
                x_base + b * BLOCK_SIZE_K * stride_xk + a_off,
                tlx.local_view(buffers_A_top, b), mask=a_mask_t, other=0.0,
            )
            tlx.async_load(
                x_base + HALF_M * stride_xm + b * BLOCK_SIZE_K * stride_xk + a_off,
                tlx.local_view(buffers_A_bot, b), mask=a_mask_b, other=0.0,
            )
            tlx.async_load_commit_group()
        tlx.async_load_wait_group(0)
        _workgroup_barrier()

        # Overlapped two-dot-pair epilogue over the 2 K-tiles (no refill).
        ce = (num_k_tiles - 2) % 2
        oe = 1 - ce
        b_cur = tlx.local_load(tlx.local_view(buffers_B, 0), token=None)
        at_cur = tlx.local_load(tlx.local_view(buffers_A_top, 0), token=None)
        ab_cur = tlx.local_load(tlx.local_view(buffers_A_bot, ce), token=None)
        _workgroup_barrier()
        acc_top = tl.dot(at_cur, b_cur, acc_top)
        acc_bot = tl.dot(ab_cur, b_cur, acc_bot)
        _workgroup_barrier()
        b_cur = tlx.local_load(tlx.local_view(buffers_B, oe), token=None)
        at_cur = tlx.local_load(tlx.local_view(buffers_A_top, oe), token=None)
        ab_cur = tlx.local_load(tlx.local_view(buffers_A_bot, oe), token=None)
        acc_top = tl.dot(at_cur, b_cur, acc_top)
        acc_bot = tl.dot(ab_cur, b_cur, acc_bot)

        _workgroup_barrier()
        yt = acc_top.to(y_ptr.dtype.element_ty)
        yb = acc_bot.to(y_ptr.dtype.element_ty)
        _workgroup_barrier()

        tl.store(y_ptr + row_t[:, None] * stride_ym + offs_n[None, :] * stride_yn,
                    yt, mask=smask_mt & mask_n_col)
        tl.store(y_ptr + row_b[:, None] * stride_ym + offs_n[None, :] * stride_yn,
                    yb, mask=smask_mb & mask_n_col)

        tile_idx += NUM_CUS
        _workgroup_barrier()


def amd_grouped_gemm_fprop(x, w, *, y=None, num_cus=None):
    """Y = X @ W^T. x [GM,K], w [1,N,K], y [GM,N]."""
    GM, K = x.shape
    _, N, K_w = w.shape
    assert K == K_w
    assert x.stride(1) == 1 and x.stride(0) == K, "x must be row-contiguous"
    assert w.stride(2) == 1 and w.stride(1) == K, "w must be row-contiguous"
    if y is None:
        out_dtype = torch.float32 if os.environ.get("STORE_FP32") == "1" else x.dtype
        y = torch.empty((GM, N), device=x.device, dtype=out_dtype)
        if os.environ.get("GLOCAL_ZERO_Y") == "1":
            y.zero_()  # so the "written" dump shows only live stores, not garbage
    if num_cus is None:
        num_cus = _num_cus()
    _amd_grouped_gemm_fprop_kernel[(num_cus,)](
        x, w, y, GM, N, K,
        x.stride(0), x.stride(1), w.stride(1), w.stride(2),
        y.stride(0), y.stride(1), num_cus,
        BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=32, NUM_STAGES=2,
        num_warps=8, matrix_instr_nonkdim=16, waves_per_eu=0,
    )
    return y


def _make_inputs():
    M = int(os.environ.get("REPRO_M", str(32768)))
    N = int(os.environ.get("REPRO_N", str(8192 * 32)))
    K = int(os.environ.get("REPRO_K", str(4096)))
    torch.manual_seed(0)
    x = torch.rand(M, K, device="cuda", dtype=torch.bfloat16).contiguous()
    w = torch.randn(1, N, K, device="cuda", dtype=torch.bfloat16).contiguous()
    print(f"repro M={M} N={N} K={K}  x={tuple(x.shape)} w={tuple(w.shape)}")
    return x, w


def _run(n_runs: int) -> None:
    # GLOCAL_DUMP_POS=invalid|written -> also print unique in-tile (row%256,
    # col%256) positions of invalid (huge/nan) resp. nonzero elements.  Pair
    # with GLOCAL_ZERO_Y=1 so "written" = exactly what the live store(s) wrote.
    dump = os.environ.get("GLOCAL_DUMP_POS")
    x, w = _make_inputs()
    for r in range(n_runs):
        y = amd_grouped_gemm_fprop(x, w)
        yf = y.float()
        invalid = (yf.abs() > 1e30) | ~torch.isfinite(yf)
        nhuge = int(invalid.sum())
        print(f"  run {r}: nhuge={nhuge}")
        if dump:
            mask = invalid if dump == "invalid" else (yf != 0)
            idx = mask.nonzero(as_tuple=False)
            rc = torch.stack([idx[:, 0] % 256, idx[:, 1] % 256], dim=1)
            pairs = sorted(torch.unique(rc, dim=0).tolist())
            print(f"POS[{dump}] n={len(pairs)}: "
                  + " ".join(f"{a},{b}" for a, b in pairs))


def _repro() -> None:
    _run(int(os.environ.get("N_RUNS", "4")))


# ---------------------------------------------------------------------------
# GCN rewrite harness: run -> capture amdgcn -> string-rewrite -> re-run.
#
# `make_hsaco` only *assembles* the amdgcn, it does not re-schedule, so editing
# the dumped (post-schedule) asm changes exactly what you edit and nothing else.
# Author a rewrite in REWRITES below, then run with `REWRITE=<name>`.  The
# harness (1) compiles once with dumping on, (2) applies your rewrite to the
# dumped asm, (3) drops it into the Triton override dir under the matching
# src-hash, and (4) recompiles+reruns from the edited asm.  Each phase runs in a
# fresh subprocess so the JIT/on-disk caches and knob env start clean.
# ---------------------------------------------------------------------------
_GCN_DUMP_DIR = "/tmp/glocal_gcn/dump"
_GCN_OVR_DIR = "/tmp/glocal_gcn/override"
_GCN_LATEST_OUT = os.environ.get("GCN_OUT", "/tmp/glocal_gcn/latest.amdgcn")
_GCN_ORIG_OUT = "/tmp/glocal_gcn/original.amdgcn"


def _rw_identity(text: str) -> str:
    return text


_STORE_X4_RE = re.compile(
    r"^(\s*)global_store_dwordx4 (\S+), v\[(\d+):(\d+)\], off(?:\s+offset:(-?\d+))?\s*$"
)


def _split_dwordx4(text: str) -> str:
    """Split every 128-bit global_store_dwordx4 into two 64-bit global_store_dwordx2
    halves -- identical addresses and data, only the transaction width changes."""
    out = []
    for l in text.splitlines():
        m = _STORE_X4_RE.match(l)
        if not m:
            out.append(l)
            continue
        ind, addr, lo, hi = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        base = int(m.group(5)) if m.group(5) else 0
        lo_off = f" offset:{base}" if base else ""
        out.append(f"{ind}global_store_dwordx2 {addr}, v[{lo}:{lo + 1}], off{lo_off}")
        out.append(f"{ind}global_store_dwordx2 {addr}, v[{lo + 2}:{hi}], off offset:{base + 8}")
    return "\n".join(out) + "\n"


def _rw_split_dwordx4(text: str) -> str:
    """dwordx4-hypothesis test: split each dwordx4 store into two dwordx2 halves,
    then drain outstanding stores with a single s_waitcnt vmcnt(0) before program
    exit.  If the corruption is a dwordx4-width problem this should eliminate it."""
    out = []
    for l in _split_dwordx4(text).splitlines():
        if l.strip().startswith("s_endpgm"):
            out.append("\ts_waitcnt vmcnt(0)")
        out.append(l)
    return "\n".join(out) + "\n"


def _rw_split_dwordx4_no_wait(text: str) -> str:
    """Width isolation: split dwordx4 -> two dwordx2, with no added vmcnt drain."""
    return _split_dwordx4(text)


REWRITES = {
    "identity": _rw_identity,
    "split_dwordx4": _rw_split_dwordx4,
    "split_dwordx4_no_wait": _rw_split_dwordx4_no_wait,
}


def _gcn_experiment() -> None:
    # REWRITE may be a comma-separated pipeline applied in order, e.g.
    #   REWRITE=identity,split_dwordx4
    names = [n for n in os.environ["REWRITE"].split(",") if n]
    for n in names:
        if n not in REWRITES:
            raise SystemExit(f"unknown rewrite '{n}'; known: {sorted(REWRITES)}")
    shutil.rmtree(_GCN_DUMP_DIR, ignore_errors=True)
    shutil.rmtree(_GCN_OVR_DIR, ignore_errors=True)

    base = dict(os.environ, GLOCAL_WORKER="1", TRITON_ALWAYS_COMPILE="1")

    print("[gcn] phase 1/3: compile + dump amdgcn", flush=True)
    cap = dict(base, TRITON_KERNEL_DUMP="1", TRITON_DUMP_DIR=_GCN_DUMP_DIR, N_RUNS="1")
    subprocess.run([sys.argv[0]], env=cap, check=True)

    dumped = glob.glob(f"{_GCN_DUMP_DIR}/*/*.amdgcn")
    assert len(dumped) == 1, f"expected one amdgcn, got {dumped}"
    src = dumped[0]
    hashdir = os.path.basename(os.path.dirname(src))
    fname = os.path.basename(src)
    orig = pathlib.Path(src).read_text()
    os.makedirs(os.path.dirname(_GCN_ORIG_OUT), exist_ok=True)
    pathlib.Path(_GCN_ORIG_OUT).write_text(orig)

    print(f"[gcn] phase 2/3: rewrite pipeline {names}", flush=True)
    text = orig
    for n in names:
        before = text
        text = REWRITES[n](text)
        co = collections.Counter(before.splitlines())
        cn = collections.Counter(text.splitlines())
        removed = sum((co - cn).values())
        added = sum((cn - co).values())
        print(f"[gcn]   {n}: -{removed}/+{added} lines "
              f"({len(before.splitlines())} -> {len(text.splitlines())})", flush=True)
    pathlib.Path(_GCN_LATEST_OUT).write_text(text)
    print(f"[gcn] overridden asm -> {_GCN_LATEST_OUT}  (original -> {_GCN_ORIG_OUT})",
          flush=True)

    dst = os.path.join(_GCN_OVR_DIR, hashdir)
    os.makedirs(dst, exist_ok=True)
    pathlib.Path(os.path.join(dst, fname)).write_text(text)

    print("[gcn] phase 3/3: rerun from edited amdgcn", flush=True)
    rr = dict(base, TRITON_KERNEL_OVERRIDE="1", TRITON_OVERRIDE_DIR=_GCN_OVR_DIR,
              N_RUNS=os.environ.get("N_RUNS", "2"))
    subprocess.run([sys.argv[0]], env=rr, check=True)


if __name__ == "__main__":
    if os.environ.get("GLOCAL_WORKER") == "1":
        _run(int(os.environ.get("N_RUNS", "2")))
    elif os.environ.get("REWRITE"):
        _gcn_experiment()
    else:
        _repro()
