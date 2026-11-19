"""MXFP6 GEMM backend for gfx950 (CDNA4): ``a @ b.T`` as BF16.

``a`` is E2M3 and ``b`` is E3M2, both dense-packed uint8 (groups of
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

from ..._catalog import InvalidInput


@triton.jit
def _fp6_codes_from_bytes(b0, b1, b2):
    v0 = b0 & 63
    v1 = ((b0 >> 6) & 3) | ((b1 & 15) << 2)
    v2 = ((b1 >> 4) & 15) | ((b2 & 3) << 4)
    v3 = (b2 >> 2) & 63
    return v0, v1, v2, v3


@triton.jit
def _load_tile(a_bptrs, b_bptrs, as_ptrs, bs_ptrs, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
               BLOCK_SIZE_K: tl.constexpr):
    a0 = tl.load(a_bptrs)
    a1 = tl.load(a_bptrs + 1)
    a2 = tl.load(a_bptrs + 2)
    av0, av1, av2, av3 = _fp6_codes_from_bytes(a0, a1, a2)
    a_codes = tl.reshape(tl.join(tl.join(av0, av1), tl.join(av2, av3)), (BLOCK_SIZE_M, BLOCK_SIZE_K)).to(tl.uint8)
    b0 = tl.load(b_bptrs)
    b1 = tl.load(b_bptrs + 1)
    b2 = tl.load(b_bptrs + 2)
    bv0, bv1, bv2, bv3 = _fp6_codes_from_bytes(b0, b1, b2)
    b_codes = tl.reshape(tl.permute(tl.join(tl.join(bv0, bv1), tl.join(bv2, bv3)), (0, 2, 3, 1)),
                         (BLOCK_SIZE_K, BLOCK_SIZE_N)).to(tl.uint8)
    sa = tl.load(as_ptrs)
    sb = tl.load(bs_ptrs)
    return a_codes, b_codes, sa, sb


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
):
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
    offs_t = tl.arange(0, BLOCK_SIZE_K // 4)
    offs_sk = tl.arange(0, BLOCK_SIZE_K // 32)

    a_bptrs = a_ptr + offs_m[:, None] * K + (3 * offs_t)[None, :]
    b_bptrs = b_ptr + offs_n[None, :] * K + (3 * offs_t)[:, None]
    as_ptrs = as_ptr + offs_m[:, None] * (K // 32) + offs_sk[None, :]
    bs_ptrs = bs_ptr + offs_n[:, None] * (K // 32) + offs_sk[None, :]

    # Manual staging: at most two tiles live (the compiler pipeliner
    # kept three and spilled), so compiler pipelining is off.
    a_codes, b_codes, sa, sb = _load_tile(a_bptrs, b_bptrs, as_ptrs, bs_ptrs, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Manual unroll x2: two staged steps per body, plus at most one
    # remainder step; the final tile dots after the loop.
    for _ in tl.range(1, K // BLOCK_SIZE_K - 1, 2, num_stages=1):
        a_bptrs += BLOCK_SIZE_K * 3 // 4
        b_bptrs += BLOCK_SIZE_K * 3 // 4
        as_ptrs += BLOCK_SIZE_K // 32
        bs_ptrs += BLOCK_SIZE_K // 32
        na_codes, nb_codes, nsa, nsb = _load_tile(a_bptrs, b_bptrs, as_ptrs, bs_ptrs, BLOCK_SIZE_M, BLOCK_SIZE_N,
                                                  BLOCK_SIZE_K)
        acc = tl.dot_scaled(a_codes, sa, "e2m3", b_codes, sb, "e3m2", acc)
        a_codes, b_codes, sa, sb = na_codes, nb_codes, nsa, nsb
        a_bptrs += BLOCK_SIZE_K * 3 // 4
        b_bptrs += BLOCK_SIZE_K * 3 // 4
        as_ptrs += BLOCK_SIZE_K // 32
        bs_ptrs += BLOCK_SIZE_K // 32
        na_codes, nb_codes, nsa, nsb = _load_tile(a_bptrs, b_bptrs, as_ptrs, bs_ptrs, BLOCK_SIZE_M, BLOCK_SIZE_N,
                                                  BLOCK_SIZE_K)
        acc = tl.dot_scaled(a_codes, sa, "e2m3", b_codes, sb, "e3m2", acc)
        a_codes, b_codes, sa, sb = na_codes, nb_codes, nsa, nsb
    if (K // BLOCK_SIZE_K - 1) % 2 == 1:
        a_bptrs += BLOCK_SIZE_K * 3 // 4
        b_bptrs += BLOCK_SIZE_K * 3 // 4
        as_ptrs += BLOCK_SIZE_K // 32
        bs_ptrs += BLOCK_SIZE_K // 32
        na_codes, nb_codes, nsa, nsb = _load_tile(a_bptrs, b_bptrs, as_ptrs, bs_ptrs, BLOCK_SIZE_M, BLOCK_SIZE_N,
                                                  BLOCK_SIZE_K)
        acc = tl.dot_scaled(a_codes, sa, "e2m3", b_codes, sb, "e3m2", acc)
        a_codes, b_codes, sa, sb = na_codes, nb_codes, nsa, nsb
    acc = tl.dot_scaled(a_codes, sa, "e2m3", b_codes, sb, "e3m2", acc)

    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16))


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
    return configs


_kernel_full = triton.autotune(configs=_full_configs(), key=["M", "N", "K"])(_gemm_mxfp6_native_kernel)


def _launch_full(a, a_scale, b, b_scale, out):
    M, K = a.shape
    N = b.shape[0]

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]), )

    _kernel_full[grid](a, b, out, a_scale, b_scale, M, N, K)
    return out


def mm_mxfp6(a, a_scale, b, b_scale, *, out=None, sf_layout="natural", space="full"):
    """``a @ b.T`` for pre-quantized MXFP6 operands on gfx950, as BF16.

    ``a`` is E2M3 and ``b`` is E3M2, both dense-packed uint8 (groups
    of 16 codes per 12 bytes, K padded to the unpacked width). Inputs
    are validated
    by ``tlx.ops.mm_mxfp6``. ``space`` is "full" (autotune, the
    default) or "heuristic" (one shape-picked config). Only the
    "natural" scale layout is supported on this backend.
    """
    if sf_layout != "natural":
        raise InvalidInput(f"gfx950 mm_mxfp6 only supports sf_layout='natural'; got {sf_layout!r}")
    if space not in ("heuristic", "full"):
        raise InvalidInput(f"unsupported space {space!r}; expected 'heuristic' or 'full'")
    M, K = a.shape
    N = b.shape[0]
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    with torch.cuda.device(a.device):
        if space == "heuristic":
            # Start from the BF16 default tile (256x256x64); the
            # 128-wide K tile overflowed registers in this kernel.
            # The kernel has no bounds masking, so the 256-wide M/N
            # tile is only valid when it divides M/N; small shapes
            # (e.g. the 128 ones test) use a 128-wide tile at the
            # same K=64 to stay exact.
            if M % 256 == 0 and N % 256 == 0:
                block_m, block_n = 256, 256
            else:
                block_m, block_n = 128, 128
            meta = {
                "BLOCK_SIZE_M": block_m,
                "BLOCK_SIZE_N": block_n,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
                "num_warps": 8,
            }
            grid = (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]), )
            _gemm_mxfp6_native_kernel[grid](
                a,
                b,
                out,
                a_scale,
                b_scale,
                M,
                N,
                K,
                BLOCK_SIZE_M=meta["BLOCK_SIZE_M"],
                BLOCK_SIZE_N=meta["BLOCK_SIZE_N"],
                BLOCK_SIZE_K=meta["BLOCK_SIZE_K"],
                GROUP_SIZE_M=meta["GROUP_SIZE_M"],
                num_warps=meta["num_warps"],
                num_stages=1,
            )
            return out
        return _launch_full(a, a_scale, b, b_scale, out)
