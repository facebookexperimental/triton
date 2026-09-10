"""Frozen gfx942 fast paths shared by :func:`tlx.ops.mm` and ``addmm``.

The five configurations below specialize BF16 ``A[M, K] @ B[K, N]`` with a
row-major A and column-major B.  They are implementation details of the generic
ops: calls that do not match this table continue through ``gfx942.py``'s normal
heuristic/autotuning path.
"""

import torch

import triton
import triton.language as tl

# Cache policy understood by _policy_load. Keeping the integer in the
# kernel signature makes every selected policy a compile-time branch.
_CACHE_DEFAULT = 0
_CACHE_CA_EVICT_LAST = 2
_CACHE_CA_EVICT_FIRST = 3
_CACHE_EVICT_LAST = 5


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
def tuned_gemm_kernel_gfx942(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
    ADD_BIAS: tl.constexpr,
    A_POLICY: tl.constexpr,
    B_POLICY: tl.constexpr,
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

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int32)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)) % N
    offs_k = tl.arange(0, BLOCK_K).to(tl.int32)
    reg_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    reg_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        k = k_idx * BLOCK_K
        a_ptrs = a_ptr + reg_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        b_ptrs = b_ptr + (k + offs_k[:, None]) * stride_bk + reg_n[None, :] * stride_bn
        a = _policy_load(a_ptrs, offs_k[None, :] < K - k, K % BLOCK_K == 0, A_POLICY)
        b = _policy_load(b_ptrs, offs_k[:, None] < K - k, K % BLOCK_K == 0, B_POLICY)
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
    tl.store(c_ptr + idx_m * N + idx_n, acc, mask=mask)


# Public, reviewable record of the five selected configurations. The keys are
# (M, N, K) for A[M, K] @ B[K, N]. Backend flags are separated from kernel
# meta-parameters by launch_tuned so they are never forwarded as kernel args.
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


def launch_tuned(a, b, bias, bias_strides, out, selected):
    """Launch one selected configuration into a validated output tensor."""
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
    )
    grid = (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]), )
    tuned_gemm_kernel_gfx942[grid](*args, ADD_BIAS=bias is not None, **meta, **backend)
