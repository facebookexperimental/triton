"""TLX fused addmm for AMD MI300X (gfx942 / CDNA3): ``out = bias + a @ b``.

The gfx950 sibling, ``amd_addmm_gfx950.py``, does not run here. It is built on
``gfx9_gemm/inter_wave/a16w16``, whose direct-to-LDS (``buffer_load_to_local``)
loads fail to legalize on CDNA3, and its tile overflows the LDS budget besides
(measured: ``OutOfResources ... Required: 67456, Hardware limit: 65536``).

So this is the ``amd_gemm_gfx942`` kernel with the bias folded into the
epilogue: the same register-staged operand path (``tl.load`` -> VGPR ->
``tlx.local_store`` -> LDS -> ``tlx.local_load`` -> MFMA), the same autotuned
tile and LDS ring depth, the same 64 KB CDNA3 budget pruning. See that file for
why CDNA3 wants register staging rather than direct-to-LDS.

Fusing the bias is free: it is one ``tl.load`` of a row (or tile) plus an add on
the accumulator that is already in registers, against a separate kernel launch
plus a full round trip of the M*N output through HBM.

``a`` is (M, K) row-major, ``b`` is (K, N) -- both in the layouts
``torch.addmm`` itself takes, no transpose required. ``bias`` may be 1-D ``(N,)``
(the common Linear case, broadcast down the rows) or a full ``(M, N)``.

Exposes ``addmm`` for the correctness suite (``testing/test_correctness.py``) and
the perf script (``testing/test_amd_addmm_gfx942_perf.py``, which compares
against aten's ``torch.addmm``).
"""

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from triton.language.extra.tlx.tutorials.amd_gemm_gfx942 import (
    CDNA3_LDS_BYTES,
    NUM_XCDS,
    _xcd_remap,
    lds_bytes,
)


@triton.jit
def addmm_kernel_gfx942(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_biasm,
    stride_biasn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    """C = bias + A @ B, register-staged LDS ring (see amd_gemm_gfx942)."""
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    if NUM_XCDS != 1:
        pid = _xcd_remap(pid, num_pid_m * num_pid_n, NUM_XCDS)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    K_ITERS = tl.cdiv(K, BLOCK_K)

    smem_a = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), NUM_BUFFERS)
    smem_b = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(b_ptr), NUM_BUFFERS)

    for i in tl.range(0, NUM_BUFFERS, loop_unroll_factor=NUM_BUFFERS):
        a_reg = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_K)
        b_reg = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_K)
        tlx.local_store(tlx.local_view(smem_a, i), a_reg)
        tlx.local_store(tlx.local_view(smem_b, i), b_reg)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in tl.range(NUM_BUFFERS, K_ITERS, num_stages=1):
        buf = k % NUM_BUFFERS
        a_reg = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K)
        b_reg = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K)

        a_tile = tlx.local_load(tlx.local_view(smem_a, buf))
        b_tile = tlx.local_load(tlx.local_view(smem_b, buf))
        acc = tl.dot(a_tile, b_tile, acc)

        tlx.local_store(tlx.local_view(smem_a, buf), a_reg)
        tlx.local_store(tlx.local_view(smem_b, buf), b_reg)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    for i in tl.range(0, NUM_BUFFERS, loop_unroll_factor=NUM_BUFFERS):
        buf = (K_ITERS + i) % NUM_BUFFERS
        a_tile = tlx.local_load(tlx.local_view(smem_a, buf))
        b_tile = tlx.local_load(tlx.local_view(smem_b, buf))
        acc = tl.dot(a_tile, b_tile, acc)

    # Epilogue: the only difference from amd_gemm_gfx942. The accumulator is
    # already in registers, so the bias costs one masked load and one add. A 1-D
    # (N,) bias arrives with stride_biasm == 0, which broadcasts it down the rows
    # for free -- no separate code path.
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    bias_ptrs = bias_ptr + offs_cm[:, None] * stride_biasm + offs_cn[None, :] * stride_biasn
    bias = tl.load(bias_ptrs, mask=mask)
    acc += bias.to(tl.float32)

    c = acc.to(tlx.dtype_of(c_ptr))
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=mask)


def _configs():
    """Same tile space as amd_gemm_gfx942 -- the hot loop is identical."""
    tiles = [
        (64, 64, 64, 4),
        (128, 128, 32, 4),
        (128, 128, 32, 8),
        (128, 128, 64, 8),
        (256, 128, 32, 8),
        (128, 256, 32, 8),
        (256, 256, 32, 8),
    ]
    return [
        triton.Config(
            {
                "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": gm, "NUM_BUFFERS": nb, "NUM_XCDS": NUM_XCDS,
                "waves_per_eu": 0
            },
            num_warps=warps,
            num_stages=1,
        ) for (bm, bn, bk, warps) in tiles for gm in (4, 8) for nb in (1, 2, 3)
    ]


def _prune_configs(configs, named_args, **kwargs):
    """Drop tiles that cannot run on this shape before anything is compiled."""
    K = named_args["K"]
    elem_bytes = named_args["a_ptr"].element_size()
    kept = []
    for config in configs:
        bm = config.kwargs["BLOCK_M"]
        bn = config.kwargs["BLOCK_N"]
        bk = config.kwargs["BLOCK_K"]
        nb = config.kwargs["NUM_BUFFERS"]
        if triton.cdiv(K, bk) < nb:
            continue
        if lds_bytes(bm, bn, bk, nb, elem_bytes) > CDNA3_LDS_BYTES:
            continue
        kept.append(config)
    if not kept:
        raise RuntimeError(f"No config fits K={K} within the {CDNA3_LDS_BYTES} B gfx942 LDS budget")
    return kept


addmm_kernel_gfx942 = triton.autotune(
    configs=_configs(),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_configs},
)(addmm_kernel_gfx942)


def addmm(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """out = bias + a @ b on AMD MI300X (gfx942).

    ``bias`` is ``(N,)`` or broadcastable to ``(M, N)``; ``a`` is (M, K) row-major
    and ``b`` is (K, N) -- the same layouts ``torch.addmm`` accepts.
    """
    assert a.shape[1] == b.shape[0], f"K mismatch: A={tuple(a.shape)}, B={tuple(b.shape)}"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert a.dtype == b.dtype == bias.dtype, "bias, A and B must have the same dtype"

    M, K = a.shape
    _, N = b.shape

    bias_view = bias.expand(M, N) if bias.ndim == 1 else bias.broadcast_to(M, N)
    assert bias_view.shape == (M, N), f"bias {tuple(bias.shape)} is not broadcastable to {(M, N)}"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )
    addmm_kernel_gfx942[grid](
        a,
        b,
        bias_view,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        bias_view.stride(0),
        bias_view.stride(1),
        c.stride(0),
        c.stride(1),
        matrix_instr_nonkdim=16,
    )
    return c
