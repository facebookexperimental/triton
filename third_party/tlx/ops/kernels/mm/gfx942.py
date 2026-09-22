"""Shared MI300X (gfx942/CDNA3) GEMM implementation for ``mm`` and ``addmm``.

One direct-load kernel serves both fundamental operations. Five BF16
row-major-A/column-major-B shapes have frozen configurations; other supported
shapes and layouts use the same kernel through a compact heuristic or autotune
space.
"""

import functools

import torch

import triton
import triton.language as tl

# Cache policy understood by _policy_load. Keeping the integer in the
# kernel signature makes every selected policy a compile-time branch.
_CACHE_DEFAULT = 0
_CACHE_CA_EVICT_LAST = 2
_CACHE_CA_EVICT_FIRST = 3
_CACHE_EVICT_LAST = 5

# Keep the steady-state K loop unmasked and handle an uneven final tile once.
_PEEL_K_TAIL = True


@triton.jit
def _policy_load(ptrs, mask, even_k: tl.constexpr, policy: tl.constexpr):
    if policy == 2:
        return tl.load(ptrs, cache_modifier=".ca", eviction_policy="evict_last") if even_k else tl.load(
            ptrs, mask=mask, other=0.0, cache_modifier=".ca", eviction_policy="evict_last")
    if policy == 3:
        return tl.load(ptrs, cache_modifier=".ca", eviction_policy="evict_first") if even_k else tl.load(
            ptrs, mask=mask, other=0.0, cache_modifier=".ca", eviction_policy="evict_first")
    if policy == 5:
        return tl.load(ptrs, eviction_policy="evict_last") if even_k else tl.load(ptrs, mask=mask, other=0.0,
                                                                                  eviction_policy="evict_last")
    return tl.load(ptrs) if even_k else tl.load(ptrs, mask=mask, other=0.0)


@triton.jit
def matmul_kernel_gfx942(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bias_m: tl.constexpr,
    stride_bias_n: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    A_POLICY: tl.constexpr,
    B_POLICY: tl.constexpr,
    PEEL_K_TAIL: tl.constexpr,
    SPLIT_M_128_32: tl.constexpr = False,
):
    """Register-staged GEMM with per-operand cache and XCD policy."""
    pid = tl.program_id(0).to(tl.int32)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    grid_mn = grid_m * grid_n

    # Stripe complete chunks over the eight XCDs.  Leave a short tail in its
    # original order so no remapped pid can escape the output-tile grid.
    if NUM_XCDS != 1:
        aligned = (grid_mn // (NUM_XCDS * XCD_CHUNK)) * (NUM_XCDS * XCD_CHUNK)
        if pid < aligned:
            xcd = pid % NUM_XCDS
            local_pid = pid // NUM_XCDS
            pid = ((local_pid // XCD_CHUNK) * NUM_XCDS * XCD_CHUNK + xcd * XCD_CHUNK + local_pid % XCD_CHUNK)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    if SPLIT_M_128_32:
        # Triton tensor dimensions must be powers of two. Represent BM=160 as
        # two panels while sharing the B tile and K loop.
        tl.static_assert(BLOCK_M == 160)
        tl.static_assert(K % BLOCK_K == 0)
        base_m = pid_m * BLOCK_M
        base_n = pid_n * BLOCK_N
        offs_m0 = (base_m + tl.arange(0, 128).to(tl.int32)) % M
        offs_m1 = (base_m + 128 + tl.arange(0, 32).to(tl.int32)) % M
        offs_n = (base_n + tl.arange(0, BLOCK_N).to(tl.int32)) % N
        offs_k = tl.arange(0, BLOCK_K).to(tl.int32)

        acc0 = tl.zeros((128, BLOCK_N), tl.float32)
        acc1 = tl.zeros((32, BLOCK_N), tl.float32)
        for k in range(0, K, BLOCK_K):
            b_ptrs = b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
            a0_ptrs = a_ptr + offs_m0[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
            a1_ptrs = a_ptr + offs_m1[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
            b = _policy_load(b_ptrs, offs_k[:, None] < K - k, True, B_POLICY)
            a0 = _policy_load(a0_ptrs, offs_k[None, :] < K - k, True, A_POLICY)
            a1 = _policy_load(a1_ptrs, offs_k[None, :] < K - k, True, A_POLICY)
            acc0 = tl.dot(a0, b, acc0, allow_tf32=False, out_dtype=tl.float32)
            acc1 = tl.dot(a1, b, acc1, allow_tf32=False, out_dtype=tl.float32)

        rows0 = base_m + tl.arange(0, 128).to(tl.int32)
        rows1 = base_m + 128 + tl.arange(0, 32).to(tl.int32)
        cols = base_n + tl.arange(0, BLOCK_N).to(tl.int32)
        idx_n = cols[None, :]
        idx_m0 = rows0[:, None]
        idx_m1 = rows1[:, None]
        mask0 = (idx_m0 < M) & (idx_n < N)
        mask1 = (idx_m1 < M) & (idx_n < N)
        if ADD_BIAS:
            bias0 = tl.load(
                bias_ptr + idx_m0 * stride_bias_m + idx_n * stride_bias_n,
                mask=mask0,
                eviction_policy="evict_last",
            )
            bias1 = tl.load(
                bias_ptr + idx_m1 * stride_bias_m + idx_n * stride_bias_n,
                mask=mask1,
                eviction_policy="evict_last",
            )
            acc0 += bias0.to(tl.float32)
            acc1 += bias1.to(tl.float32)
        tl.store(c_ptr + idx_m0 * stride_cm + idx_n * stride_cn, acc0, mask=mask0)
        tl.store(c_ptr + idx_m1 * stride_cm + idx_n * stride_cn, acc1, mask=mask1)
    else:
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int32)) % M
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)) % N
        offs_k = tl.arange(0, BLOCK_K).to(tl.int32)
        reg_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
        reg_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        even_k = K % BLOCK_K == 0
        if PEEL_K_TAIL:
            k_main = K if even_k else (K // BLOCK_K) * BLOCK_K
            for k in range(0, k_main, BLOCK_K):
                a_ptrs = a_ptr + reg_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
                b_ptrs = b_ptr + (k + offs_k[:, None]) * stride_bk + reg_n[None, :] * stride_bn
                a = _policy_load(a_ptrs, offs_k[None, :] < K - k, True, A_POLICY)
                b = _policy_load(b_ptrs, offs_k[:, None] < K - k, True, B_POLICY)
                acc = tl.dot(a, b, acc, allow_tf32=False, out_dtype=tl.float32)
            if not even_k:
                a_ptrs = a_ptr + reg_m[:, None] * stride_am + (k_main + offs_k[None, :]) * stride_ak
                b_ptrs = b_ptr + (k_main + offs_k[:, None]) * stride_bk + reg_n[None, :] * stride_bn
                tail = offs_k < K - k_main
                a = _policy_load(a_ptrs, tail[None, :], False, A_POLICY)
                b = _policy_load(b_ptrs, tail[:, None], False, B_POLICY)
                acc = tl.dot(a, b, acc, allow_tf32=False, out_dtype=tl.float32)
        else:
            for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
                k = k_idx * BLOCK_K
                a_ptrs = a_ptr + reg_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
                b_ptrs = b_ptr + (k + offs_k[:, None]) * stride_bk + reg_n[None, :] * stride_bn
                a = _policy_load(a_ptrs, offs_k[None, :] < K - k, even_k, A_POLICY)
                b = _policy_load(b_ptrs, offs_k[:, None] < K - k, even_k, B_POLICY)
                acc = tl.dot(a, b, acc, allow_tf32=False, out_dtype=tl.float32)

        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int32)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)
        idx_m = rows[:, None]
        idx_n = cols[None, :]
        mask = (idx_m < M) & (idx_n < N)
        if ADD_BIAS:
            bias = tl.load(
                bias_ptr + idx_m * stride_bias_m + idx_n * stride_bias_n,
                mask=mask,
                eviction_policy="evict_last",
            )
            acc += bias.to(tl.float32)
        tl.store(c_ptr + idx_m * stride_cm + idx_n * stride_cn, acc, mask=mask)


# Public, reviewable record of the five selected configurations. The keys are
# (M, N, K) for A[M, K] @ B[K, N]. Backend flags are separated from kernel
# meta-parameters by ``_launch_config`` so they are never forwarded as kernel
# arguments.
TUNED_CONFIGS = {
    (819200, 1024, 192): {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 32,
        "GROUP_M": 32,
        "NUM_XCDS": 8,
        "XCD_CHUNK": 4,
        "A_POLICY": _CACHE_CA_EVICT_FIRST,
        "B_POLICY": _CACHE_DEFAULT,
        "matrix_instr_nonkdim": 16,
        "waves_per_eu": 0,
        "kpack": 2,
        "num_warps": 4,
        "num_stages": 2,
        "LLVM_SCHED_STRATEGY": "max-memory-clause",
    },
    (4096, 1894, 242432): {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_K": 256,
        "GROUP_M": 32,
        "NUM_XCDS": 8,
        "XCD_CHUNK": 8,
        "A_POLICY": _CACHE_EVICT_LAST,
        "B_POLICY": _CACHE_DEFAULT,
        "matrix_instr_nonkdim": 16,
        "waves_per_eu": 0,
        "kpack": 1,
        "num_warps": 8,
        "num_stages": 2,
        "DISABLE_AGPR": True,
        "REVERSE_LOCAL_ASSIGNMENT": True,
        "SINK_INSTS_TO_AVOID_SPILLS": True,
        "REGCLASS_PRIORITY": False,
        "DISABLE_HIGH_RP_RESCHEDULE": False,
    },
    (1024, 6144, 20480): {
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 64,
        "GROUP_M": 2,
        "NUM_XCDS": 8,
        "XCD_CHUNK": 8,
        "A_POLICY": _CACHE_DEFAULT,
        "B_POLICY": _CACHE_EVICT_LAST,
        "matrix_instr_nonkdim": 16,
        "waves_per_eu": 0,
        "kpack": 1,
        "num_warps": 4,
        "num_stages": 2,
        "ENABLE_SCHED_BARRIER": True,
        "SINK_INSTS_TO_AVOID_SPILLS": True,
    },
    (2048, 25408, 10240): {
        "BLOCK_M": 256,
        "BLOCK_N": 256,
        "BLOCK_K": 64,
        "GROUP_M": 8,
        "NUM_XCDS": 8,
        "XCD_CHUNK": 16,
        "A_POLICY": _CACHE_CA_EVICT_LAST,
        "B_POLICY": _CACHE_DEFAULT,
        "matrix_instr_nonkdim": 16,
        "waves_per_eu": 1,
        "kpack": 1,
        "num_warps": 8,
        "num_stages": 2,
        "ENABLE_SCHED_BARRIER": True,
        "DISABLE_AGPR": True,
        "REGCLASS_PRIORITY": True,
        "LLVM_SCHED_STRATEGY": "iterative-ilp",
    },
    (61440, 2048, 5120): {
        "BLOCK_M": 256,
        "BLOCK_N": 256,
        "BLOCK_K": 64,
        "GROUP_M": 4,
        "NUM_XCDS": 8,
        "XCD_CHUNK": 16,
        "A_POLICY": _CACHE_EVICT_LAST,
        "B_POLICY": _CACHE_DEFAULT,
        "matrix_instr_nonkdim": 16,
        "waves_per_eu": 1,
        "kpack": 1,
        "num_warps": 8,
        "num_stages": 2,
        "ENABLE_SCHED_BARRIER": True,
        "DISABLE_AGPR": True,
    },
}

_BACKEND_OPTIONS = {
    "ENABLE_SCHED_BARRIER": "enable_sched_group_barrier_scheduler",
    "REVERSE_LOCAL_ASSIGNMENT": "reverse_local_assignment",
    "SINK_INSTS_TO_AVOID_SPILLS": "sink_insts_to_avoid_spills",
    "REGCLASS_PRIORITY": "regclass_priority_trumps_globalness",
    "DISABLE_HIGH_RP_RESCHEDULE": "disable_unclustered_high_rp_reschedule",
}


def tuned_config(a: torch.Tensor, b: torch.Tensor):
    """Return an exact-shape fast path, or ``None`` for the generic path.

    The configurations were measured with contiguous ``A[M, K]`` storage and
    a transposed contiguous ``B[K, N]`` view. Keeping the layout guard here
    prevents a shape match from silently applying a layout-specific choice to
    a different memory-access pattern.
    """
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        return None
    if a.dtype != torch.bfloat16 or b.dtype != a.dtype:
        return None
    if not a.is_contiguous() or b.stride() != (1, b.shape[0]):
        return None
    return TUNED_CONFIGS.get((a.shape[0], b.shape[1], a.shape[1]))


def _launch_config(a, b, bias, bias_strides, out, selected):
    """Launch one frozen configuration into a validated output tensor."""
    meta = dict(selected)

    backend = {}
    llvm_attrs = []
    if meta.pop("DISABLE_AGPR", False):
        llvm_attrs.append(("amdgpu-agpr-alloc", "0,0"))
    sched_strategy = meta.pop("LLVM_SCHED_STRATEGY", "")
    if sched_strategy:
        llvm_attrs.append(("amdgpu-sched-strategy", sched_strategy))
    if llvm_attrs:
        backend["llvm_fn_attrs"] = tuple(llvm_attrs)
    for key, option in _BACKEND_OPTIONS.items():
        if key in meta:
            backend[option] = meta.pop(key)

    m, k = a.shape
    n = b.shape[1]
    bias_ptr = bias if bias is not None else out
    args = (
        a,
        b,
        bias_ptr,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        bias_strides[0],
        bias_strides[1],
        out.stride(0),
        out.stride(1),
    )
    grid = (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]), )
    matmul_kernel_gfx942[grid](
        *args,
        ADD_BIAS=bias is not None,
        PEEL_K_TAIL=_PEEL_K_TAIL,
        **meta,
        **backend,
    )


def _config(block_m, block_n, block_k, group_m, num_warps, *, waves_per_eu=0, kpack=1, split_m_128_32=False):
    # This overlaps register-staged global loads; it is not an explicit
    # two-buffer LDS allocation.
    meta = {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": group_m,
        "NUM_XCDS": 8,
        "XCD_CHUNK": 8,
        "A_POLICY": _CACHE_DEFAULT,
        "B_POLICY": _CACHE_DEFAULT,
        "waves_per_eu": waves_per_eu,
        "kpack": kpack,
    }
    if split_m_128_32:
        meta["SPLIT_M_128_32"] = True
    return triton.Config(meta, num_warps=num_warps, num_stages=2)


def _configs():
    """Compact generic search space for the direct-load kernel."""
    return [
        _config(64, 64, 64, 4, 4),
        _config(64, 64, 128, 8, 8),
        _config(128, 64, 64, 4, 8),
        _config(64, 128, 64, 8, 8),
        _config(128, 128, 32, 8, 4),
        _config(128, 128, 64, 8, 8),
        _config(256, 128, 32, 8, 8),
        _config(128, 256, 32, 8, 8),
        _config(256, 256, 64, 8, 8),
    ]


CONFIGS = _configs


def _smoke_configs():
    return [_config(64, 64, 64, 4, 4), _config(128, 128, 32, 8, 4)]


SMOKE_CONFIGS = _smoke_configs


def heuristic_config(M, N, K):
    """Choose one direct-load configuration without runtime autotuning."""
    if (M, N, K) == (2048, 10240, 25408):
        return [_config(160, 512, 32, 8, 8, split_m_128_32=True)]
    if min(M, N) <= 64:
        return [_config(64, 64, 64, 4, 4)]
    if K <= 256:
        return [_config(128, 128, 32, 8, 4)]
    wide_workgroups = triton.cdiv(M, 256) * triton.cdiv(N, 256)
    if M >= 2048 and N >= 2048 and wide_workgroups >= 256:
        return [_config(256, 256, 64, 8, 8)]
    if M < N:
        return [_config(128, 256, 32, 8, 8)]
    if N < M:
        return [_config(256, 128, 32, 8, 8)]
    return [_config(128, 128, 64, 8, 8)]


@functools.lru_cache(maxsize=None)
def _tuned(space, shape=None):
    """Autotuned direct-load kernel per search space."""
    if space == "heuristic":
        configs = heuristic_config(*shape)
    else:
        configs = {"full": CONFIGS, "smoke": SMOKE_CONFIGS}[space]()
    return triton.autotune(configs=configs, key=["M", "N", "K", "ADD_BIAS", "PEEL_K_TAIL"])(matmul_kernel_gfx942)


def _validate_operands(a, b, out):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected A[M, K] and B[K, N], got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"K mismatch: A={tuple(a.shape)}, B={tuple(b.shape)}")
    if a.device.type != "cuda" or b.device != a.device:
        raise ValueError("A and B must be on the same GPU")
    if a.dtype != b.dtype:
        raise ValueError("A and B must have the same dtype")
    M, K = a.shape
    N = b.shape[1]
    if out is not None:
        if out.shape != (M, N) or out.device != a.device or out.dtype != a.dtype or not out.is_contiguous():
            raise ValueError(f"out must be a contiguous {a.dtype} tensor with shape ({M}, {N}) on A's device")
    return M, N, K


def _bias_strides(bias, M, N, a):
    if bias.device != a.device or bias.dtype != a.dtype:
        raise ValueError("input must match A's device and dtype")
    if bias.ndim == 1:
        if bias.shape[0] != N:
            raise ValueError(f"1-D addmm input must have shape ({N},), got {tuple(bias.shape)}")
        return 0, bias.stride(0)
    if bias.ndim == 2 and bias.shape[0] in (1, M) and bias.shape[1] in (1, N):
        return (0 if bias.shape[0] == 1 else bias.stride(0), 0 if bias.shape[1] == 1 else bias.stride(1))
    raise ValueError(f"addmm input with shape {tuple(bias.shape)} is not broadcastable to ({M}, {N})")


def _gemm(a, b, bias=None, *, out=None, space="heuristic"):
    M, N, K = _validate_operands(a, b, out)
    bias_strides = _bias_strides(bias, M, N, a) if bias is not None else (0, 0)
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=a.dtype)

    selected = tuned_config(a, b) if space == "heuristic" else None
    if selected is not None:
        _launch_config(a, b, bias, bias_strides, out, selected)
        return out

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )  # noqa: E731
    kernel = _tuned(space, (M, N, K) if space == "heuristic" else None)
    bias_ptr = bias if bias is not None else out
    kernel[grid](
        a,
        b,
        bias_ptr,
        out,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        bias_strides[0],
        bias_strides[1],
        out.stride(0),
        out.stride(1),
        ADD_BIAS=bias is not None,
        PEEL_K_TAIL=_PEEL_K_TAIL,
        matrix_instr_nonkdim=16,
    )
    return out


def mm(a, b, *, out=None, space="heuristic"):
    """Compute ``a @ b`` using the gfx942 direct-load GEMM kernel."""
    return _gemm(a, b, out=out, space=space)


# Compatibility entry point used by the kernel-optimization agent.
matmul = mm
