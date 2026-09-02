"""TLX batched GEMM (BMM) for AMD MI300X (gfx942 / CDNA3): ``C[i] = A[i] @ B[i]``.

The two gfx950 siblings, ``amd_bmm.py`` and ``amd_bmm_shared_a.py``, do not run
here -- both overflow the CDNA3 LDS budget (measured: ``OutOfResources ...
Required: 75968, Hardware limit: 65536``) and their fast path is the
direct-to-LDS one that CDNA3 caps at 32 bits per lane.

So this is the ``amd_gemm_gfx942`` kernel with a batch axis: the same
register-staged operand path, the same autotuned tile and LDS ring depth, the
same 64 KB budget pruning. See that file for why CDNA3 wants register staging.

Two consequences of the register path worth calling out, both simplifications
over the gfx950 kernels:

* **B can be row-major.** ``amd_bmm.py`` requires column-major B
  (``stride_bk == 1``) because that is what its direct-to-LDS loads need; a
  register-staged load does not care, so this kernel takes B exactly as
  ``torch.bmm`` hands it over and needs no companion file for the other layout.
* **Shared-A costs nothing to support.** ``a.stride(0) == 0`` (one (M, K) matrix
  broadcast over the batch, what ``make_bmm_inputs`` builds) is just a zero batch
  stride, so it falls out of generic stride arithmetic.

The batch index is the second grid axis rather than being folded into the tile
id, which keeps the XCD remap and GROUP_M swizzle operating on the M/N tile grid
exactly as they do in the single-matrix kernel.

Batch base offsets are computed in int64: ``batch * M * K`` overflows int32 well
inside the shapes this is benchmarked at. Within-tile offsets stay int32.

Exposes ``bmm`` and ``make_bmm_inputs`` for the correctness suite
(``testing/test_correctness.py``) and the perf script
(``testing/test_amd_bmm_gfx942_perf.py``, which compares against aten's
``torch.bmm``).
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
def bmm_kernel_gfx942(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    """C[i] = A[i] @ B[i], register-staged LDS ring (see amd_gemm_gfx942)."""
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    pid = tl.program_id(0)
    batch = tl.program_id(1)
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

    # int64 batch base: batch * M * K exceeds int32 at the shapes this is used
    # at. Everything below stays int32 and is added to these bases.
    batch64 = batch.to(tl.int64)
    a_base = a_ptr + batch64 * stride_ab
    b_base = b_ptr + batch64 * stride_bb
    c_base = c_ptr + batch64 * stride_cb

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_base + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_base + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

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

    c = acc.to(tlx.dtype_of(c_ptr))
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_base + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


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


bmm_kernel_gfx942 = triton.autotune(
    configs=_configs(),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _prune_configs},
)(bmm_kernel_gfx942)


def bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C[i] = A[i] @ B[i] on AMD MI300X (gfx942).

    ``a`` is (B, M, K) and ``b`` is (B, K, N), in the layouts ``torch.bmm``
    accepts. A zero batch stride on ``a`` (shared-A) is supported.
    """
    assert a.ndim == 3 and b.ndim == 3, f"expected 3-D inputs, got {a.ndim}-D and {b.ndim}-D"
    assert a.shape[0] == b.shape[0], f"batch mismatch: {a.shape[0]} vs {b.shape[0]}"
    assert a.shape[2] == b.shape[1], f"K mismatch: A={tuple(a.shape)}, B={tuple(b.shape)}"
    assert a.dtype == b.dtype, "A and B must have the same dtype"

    Bs, M, K = a.shape
    N = b.shape[2]

    c = torch.empty((Bs, M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), Bs)
    bmm_kernel_gfx942[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        a.stride(2),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        c.stride(2),
        matrix_instr_nonkdim=16,
    )
    return c


def make_bmm_inputs(B, M, N, K, device, dtype=torch.float16, seed=0):
    """Shared-A inputs, matching ``amd_bmm.make_bmm_inputs``' benchmark convention.

    One (M, K) matrix is reused across the batch via ``expand``, so
    ``a.stride(0) == 0``. hipBLASLt exploits that to read A once and keep it
    L2-resident, so benchmarking against distinct-A would flatter TLX. Unlike the
    gfx950 kernel, B is left row-major -- the register-staged loads do not need
    the column-major layout that direct-to-LDS does.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn((M, K), device=device, dtype=dtype, generator=g).unsqueeze(0).expand(B, M, K)
    b = torch.randn((B, K, N), device=device, dtype=dtype, generator=g)
    return a, b
