"""MXFP6 GEMM backend for gfx950 (CDNA4): ``a @ b.T`` as BF16.

``a_format`` and ``b_format`` independently select E2M3 or E3M2.
Both operands are dense-packed uint8 (groups of
16 codes per 12 bytes, K padded to the unpacked width; shapes count
logical values in multiples of 16). The only implementation is the
native scaled-MFMA kernel; ``space`` selects its launch config
("heuristic" single config, "full" autotuned). Only the "natural"
scale layout is supported.

Registered in ``tlx.ops`` for gfx950: the native scaled-MFMA path
produces correct numerics for K=128 tiles, so MI350 callers dispatch
here through the catalog instead of getting ``UnsupportedOp``.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from ..._catalog import InvalidInput


@triton.jit
def _fp6_codes_from_bytes(b0, b1, b2):
    v0 = b0 & 63
    v1 = ((b0 >> 6) & 3) | ((b1 & 15) << 2)
    v2 = ((b1 >> 4) & 15) | ((b2 & 3) << 4)
    v3 = (b2 >> 2) & 63
    return v0, v1, v2, v3


@triton.jit
def _load_tile(a_bptrs, b_bptrs, as_ptrs, bs_ptrs, valid_m, valid_n, K, k_offset, BLOCK_SIZE_K: tl.constexpr):
    valid_k = k_offset + tl.arange(0, BLOCK_SIZE_K) < K
    valid_sk = k_offset + 32 * tl.arange(0, BLOCK_SIZE_K // 32) < K
    a = tl.load(a_bptrs, mask=valid_m[:, None] & valid_k[None, :], other=0)
    b = tl.load(b_bptrs, mask=valid_k[:, None] & valid_n[None, :], other=0)
    # E8M0 byte 127 encodes 1.0, the neutral scale for zero-padded codes.
    unit_scale = tl.full((), 127, tl.uint8).to(tl.float8e8m0fnu, bitcast=True)
    sa = tl.load(as_ptrs, mask=valid_m[:, None] & valid_sk[None, :], other=unit_scale)
    sb = tl.load(bs_ptrs, mask=valid_n[:, None] & valid_sk[None, :], other=unit_scale)
    return a, b, sa, sb


@triton.jit
def _gemm_mxfp6_native_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    as_ptr,
    bs_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
    A_FORMAT: tl.constexpr = "e2m3",
    B_FORMAT: tl.constexpr = "e3m2",
):
    tl.static_assert(A_FORMAT == "e2m3" or A_FORMAT == "e3m2", "unsupported A_FORMAT")
    tl.static_assert(B_FORMAT == "e2m3" or B_FORMAT == "e3m2", "unsupported B_FORMAT")
    # Native scaled-MFMA path (manual 96-byte packed K, 128 logical values): load each dense 96-byte
    # tile, unpack its 128 codes in-kernel, and feed unpacked codes to
    # tl.dot_scaled; the compiler dense-packs per lane for the
    # 16x16x128 / 32x32x64 instruction. No padded storage is used.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_sk = tl.arange(0, BLOCK_SIZE_K // 32)
    valid_m = offs_m < M
    valid_n = offs_n < N
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    if SPLIT_K > 1:
        pid_k = tl.program_id(axis=1)
        first_k_tile = pid_k * num_k_tiles // SPLIT_K
        last_k_tile = (pid_k + 1) * num_k_tiles // SPLIT_K
        k_start = first_k_tile * BLOCK_SIZE_K
        num_k_tiles = last_k_tile - first_k_tile
    else:
        k_start = 0

    # Per-segment packed loads: slot j holds dense byte
    # (j // 32) * 24 + (j % 32) (24 packed bytes + 8 bytes the
    # lowering ignores per 32-code segment, matching the kWidth=32
    # fragment). No byte->code unpack / code->byte repack.
    byte_offs = k_start * 3 // 4 + (offs_k // 32) * 24 + (offs_k % 32)
    a_bptrs = a_ptr + offs_m[:, None] * K + byte_offs[None, :]
    b_bptrs = b_ptr + offs_n[None, :] * K + byte_offs[:, None]
    as_ptrs = as_ptr + offs_m[:, None] * (K // 32) + k_start // 32 + offs_sk[None, :]
    bs_ptrs = bs_ptr + offs_n[:, None] * (K // 32) + k_start // 32 + offs_sk[None, :]

    # Manual staging: at most two tiles live (the compiler pipeliner
    # kept three and spilled), so compiler pipelining is off.
    k_offset = k_start
    a_codes, b_codes, sa, sb = _load_tile(a_bptrs, b_bptrs, as_ptrs, bs_ptrs, valid_m, valid_n, K, k_offset,
                                          BLOCK_SIZE_K)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Manual unroll x2: two staged steps per body, plus at most one
    # remainder step; the final tile dots after the loop.
    for _ in tl.range(1, num_k_tiles - 1, 2, num_stages=1):
        a_bptrs += BLOCK_SIZE_K * 3 // 4
        b_bptrs += BLOCK_SIZE_K * 3 // 4
        as_ptrs += BLOCK_SIZE_K // 32
        bs_ptrs += BLOCK_SIZE_K // 32
        k_offset += BLOCK_SIZE_K
        na_codes, nb_codes, nsa, nsb = _load_tile(a_bptrs, b_bptrs, as_ptrs, bs_ptrs, valid_m, valid_n, K, k_offset,
                                                  BLOCK_SIZE_K)
        acc = tlx.dot_scaled(a_codes, sa, A_FORMAT, b_codes, sb, B_FORMAT, acc, fp6_packed=True)
        a_codes, b_codes, sa, sb = na_codes, nb_codes, nsa, nsb
        a_bptrs += BLOCK_SIZE_K * 3 // 4
        b_bptrs += BLOCK_SIZE_K * 3 // 4
        as_ptrs += BLOCK_SIZE_K // 32
        bs_ptrs += BLOCK_SIZE_K // 32
        k_offset += BLOCK_SIZE_K
        na_codes, nb_codes, nsa, nsb = _load_tile(a_bptrs, b_bptrs, as_ptrs, bs_ptrs, valid_m, valid_n, K, k_offset,
                                                  BLOCK_SIZE_K)
        acc = tlx.dot_scaled(a_codes, sa, A_FORMAT, b_codes, sb, B_FORMAT, acc, fp6_packed=True)
        a_codes, b_codes, sa, sb = na_codes, nb_codes, nsa, nsb
    if (num_k_tiles - 1) % 2 == 1:
        a_bptrs += BLOCK_SIZE_K * 3 // 4
        b_bptrs += BLOCK_SIZE_K * 3 // 4
        as_ptrs += BLOCK_SIZE_K // 32
        bs_ptrs += BLOCK_SIZE_K // 32
        k_offset += BLOCK_SIZE_K
        na_codes, nb_codes, nsa, nsb = _load_tile(a_bptrs, b_bptrs, as_ptrs, bs_ptrs, valid_m, valid_n, K, k_offset,
                                                  BLOCK_SIZE_K)
        acc = tlx.dot_scaled(a_codes, sa, A_FORMAT, b_codes, sb, B_FORMAT, acc, fp6_packed=True)
        a_codes, b_codes, sa, sb = na_codes, nb_codes, nsa, nsb
    acc = tlx.dot_scaled(a_codes, sa, A_FORMAT, b_codes, sb, B_FORMAT, acc, fp6_packed=True)

    c_ptrs = c_ptr + offs_m.to(tl.int64)[:, None] * N + offs_n[None, :]
    if SPLIT_K > 1:
        tl.store(c_ptrs + pid_k.to(tl.int64) * M * N, acc, mask=valid_m[:, None] & valid_n[None, :])
    else:
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=valid_m[:, None] & valid_n[None, :])


@triton.jit
def _reduce_split_k(partials_ptr, out_ptr, MN, SPLIT_K: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    splits = tl.arange(0, SPLIT_K)
    partials = tl.load(partials_ptr + splits.to(tl.int64)[:, None] * MN + offsets[None, :], mask=offsets[None, :] < MN,
                       other=0)
    tl.store(out_ptr + offsets, tl.sum(partials, axis=0).to(tl.bfloat16), mask=offsets < MN)


def _full_configs():
    configs = []
    for block_m, block_n, block_k in (
        (128, 128, 128),
        (128, 256, 128),
        (256, 128, 128),
    ):
        for num_stages in (1, ):
            configs.append(
                triton.Config(
                    {
                        "BLOCK_SIZE_M": block_m,
                        "BLOCK_SIZE_N": block_n,
                        "BLOCK_SIZE_K": block_k,
                        "GROUP_SIZE_M": 8,
                    },
                    num_warps=8,
                    num_stages=num_stages,
                ))
    # 128x128x128 is fastest at 4 warps (the heuristic default).
    configs.append(
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_warps=4,
            num_stages=1,
        ))
    return configs


_kernel_full = triton.autotune(configs=_full_configs(), key=["M", "N", "K", "SPLIT_K", "A_FORMAT",
                                                             "B_FORMAT"])(_gemm_mxfp6_native_kernel)


def _launch_full(a, a_scale, b, b_scale, out, split_k=1, a_format="e2m3", b_format="e3m2"):
    M, K = a.shape
    N = b.shape[0]

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]), split_k)

    _kernel_full[grid](a, b, out, a_scale, b_scale, M, N, K, SPLIT_K=split_k, A_FORMAT=a_format, B_FORMAT=b_format)
    return out


def mm_mxfp6(a, a_scale, b, b_scale, *, out=None, sf_layout="natural", space="full", split_k=1, a_format="e2m3",
             b_format="e3m2"):
    """``a @ b.T`` for pre-quantized MXFP6 operands on gfx950, as BF16.

    ``a_format`` and ``b_format`` independently select "e2m3" or "e3m2"
    (defaulting to E2M3 A and E3M2 B), both dense-packed uint8 (groups
    of 16 codes per 12 bytes, K padded to the unpacked width). Inputs
    are validated
    by ``tlx.ops.mm_mxfp6``. ``space`` is "full" (autotune, the
    default) or "heuristic" (one shape-picked config). Only the
    "natural" scale layout is supported on this backend.

    ``split_k`` partitions whole MFMA K tiles. FP32 partials are
    reduced before the single BF16 conversion.
    """
    if a_format not in ("e2m3", "e3m2") or b_format not in ("e2m3", "e3m2"):
        raise InvalidInput("a_format and b_format must each be 'e2m3' or 'e3m2'")
    if sf_layout != "natural":
        raise InvalidInput(f"gfx950 mm_mxfp6 only supports sf_layout='natural'; got {sf_layout!r}")
    if space not in ("heuristic", "full"):
        raise InvalidInput(f"unsupported space {space!r}; expected 'heuristic' or 'full'")
    M, K = a.shape
    N = b.shape[0]
    block_k = 128 if K % 128 == 0 else 64
    if not isinstance(split_k, int) or split_k not in (1, 2, 4, 8, 16) or split_k > K // block_k:
        raise InvalidInput("split_k must be one of (1, 2, 4, 8, 16) and not exceed the number of K tiles")
    if space == "full" and K % 128:
        raise InvalidInput("space='full' requires K to be a multiple of 128")
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    with torch.cuda.device(a.device):
        target = torch.empty((split_k, M, N), device=a.device, dtype=torch.float32) if split_k > 1 else out
        if space == "heuristic":
            # Default from the MI350 warp sweep: 128x128x128 at 4
            # warps spills zero and beats the 256-wide tile; K=64
            # shapes keep the 64-deep tile (whole 32x32x64 steps).
            meta = {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": block_k,
                "GROUP_SIZE_M": 8,
                "num_warps": 4,
            }
            grid = (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]), split_k)
            _gemm_mxfp6_native_kernel[grid](
                a,
                b,
                target,
                a_scale,
                b_scale,
                M,
                N,
                K,
                BLOCK_SIZE_M=meta["BLOCK_SIZE_M"],
                BLOCK_SIZE_N=meta["BLOCK_SIZE_N"],
                BLOCK_SIZE_K=meta["BLOCK_SIZE_K"],
                GROUP_SIZE_M=meta["GROUP_SIZE_M"],
                SPLIT_K=split_k,
                A_FORMAT=a_format,
                B_FORMAT=b_format,
                num_warps=meta["num_warps"],
                num_stages=1,
            )
        else:
            _launch_full(a, a_scale, b, b_scale, target, split_k, a_format, b_format)
        if split_k > 1:
            _reduce_split_k[(triton.cdiv(M * N, 1024), )](target, out, M * N, split_k, 1024)
        return out
