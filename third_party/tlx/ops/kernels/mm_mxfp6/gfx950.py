"""MXFP6 GEMM backend for gfx950 (CDNA4): ``a @ b.T`` as BF16.

``a`` is E2M3 and ``b`` is E3M2, both dense-packed uint8 (groups of
16 codes per 12 bytes, K padded to the unpacked width; shapes count
logical values in multiples of 16). The kernel is a tiled pointer GEMM over
``tl.dot_scaled``; on gfx950 scaled FP6 dots take the decompose
(upcast) path, so this backend favors correctness and shape coverage
over the native-instruction tuning of the CDNA4 HIP libraries.
Inputs are validated by ``tlx.ops.mm_mxfp6``; only the "natural"
scale layout is supported.

Not registered in ``tlx.ops`` yet: numerics are not correct at this
revision, so the catalog has no gfx950 ``mm_mxfp6`` entry and MI350
callers get ``UnsupportedOp``. Register it once the correctness fix
lands.
"""

import torch
import triton
import triton.language as tl

from ..._catalog import InvalidInput


@triton.jit
def _gemm_mxfp6_gfx950_kernel(
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
    NUM_STAGES: tl.constexpr,
):
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

    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] + offs_n[None, :] * K
    as_ptrs = as_ptr + offs_m[:, None] * (K // 32) + offs_sk[None, :]
    bs_ptrs = bs_ptr + offs_n[:, None] * (K // 32) + offs_sk[None, :]

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in tl.range(0, K // BLOCK_SIZE_K, num_stages=NUM_STAGES):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        sa = tl.load(as_ptrs)
        sb = tl.load(bs_ptrs)
        acc = tl.dot_scaled(a, sa, "e2m3", b, sb, "e3m2", acc)
        # Each K tile holds BLOCK_SIZE_K * 3 // 4 packed bytes.
        a_ptrs += BLOCK_SIZE_K * 3 // 4
        b_ptrs += BLOCK_SIZE_K * 3 // 4
        as_ptrs += BLOCK_SIZE_K // 32
        bs_ptrs += BLOCK_SIZE_K // 32

    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16))


def _heuristic_config(M, N, K):
    if N % 256 == 0:
        block_m, block_n = 128, 256
    else:
        block_m, block_n = 128, 128
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8,
        "NUM_STAGES": 3 if K >= 1024 else 2,
        "num_warps": 8,
    }


def _full_configs():
    configs = []
    for block_m, block_n, block_k in (
        (128, 128, 128),
        (128, 256, 128),
        (256, 128, 128),
        (128, 128, 256),
    ):
        for num_stages in (2, 3):
            configs.append(
                triton.Config(
                    {
                        "BLOCK_SIZE_M": block_m,
                        "BLOCK_SIZE_N": block_n,
                        "BLOCK_SIZE_K": block_k,
                        "GROUP_SIZE_M": 8,
                        "NUM_STAGES": num_stages,
                    },
                    num_warps=8,
                ))
    return configs


_kernel_full = triton.autotune(configs=_full_configs(), key=["M", "N", "K"])(_gemm_mxfp6_gfx950_kernel)


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
            meta = _heuristic_config(M, N, K)
            grid = (triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]), )
            _gemm_mxfp6_gfx950_kernel[grid](
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
                NUM_STAGES=meta["NUM_STAGES"],
                num_warps=meta["num_warps"],
            )
            return out
        return _launch_full(a, a_scale, b, b_scale, out)
