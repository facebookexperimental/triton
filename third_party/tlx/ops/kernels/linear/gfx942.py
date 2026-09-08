"""Frozen, shape-specialized Linear kernels for MI300X (gfx942 / CDNA3).

The public entry point computes ``input @ weight.T + bias`` for five exact
BF16 production shapes. Four require a contiguous vector bias and one is
bias-free. Configurations are fixed so calls never pay an autotuning cost.
Unsupported shapes fail explicitly rather than falling back to another
provider.
"""

import torch

import triton
import triton.language as tl

# Cache policy understood by _linear_policy_load. Keeping the integer in the
# kernel signature makes every selected policy a compile-time branch.
_CACHE_DEFAULT = 0
_CACHE_CA_EVICT_LAST = 2
_CACHE_CA_EVICT_FIRST = 3
_CACHE_EVICT_LAST = 5


@triton.jit
def _linear_policy_load(ptrs, mask, even_k: tl.constexpr, policy: tl.constexpr):
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
def _linear_memory_kernel(
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
        a = _linear_policy_load(a_ptrs, offs_k[None, :] < K - k, K % BLOCK_K == 0, A_POLICY)
        b = _linear_policy_load(b_ptrs, offs_k[:, None] < K - k, K % BLOCK_K == 0, B_POLICY)
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


@triton.jit
def _linear_register_kernel(
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
    ADD_BIAS: tl.constexpr,
):
    """Minimal register GEMM retained for the third production shape."""
    pid = tl.program_id(0).to(tl.int32)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    grid_mn = grid_m * grid_n

    xcd_chunk: tl.constexpr = 4
    if NUM_XCDS != 1:
        aligned = (grid_mn // (NUM_XCDS * xcd_chunk)) * (NUM_XCDS * xcd_chunk)
        if pid < aligned:
            xcd = pid % NUM_XCDS
            local_pid = pid // NUM_XCDS
            pid = ((local_pid // xcd_chunk) * NUM_XCDS * xcd_chunk + xcd * xcd_chunk + local_pid % xcd_chunk)

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
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b, allow_tf32=False, out_dtype=tl.float32)

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


# Public, reviewable record of the five selected configurations.  The keys are
# (M, N, K) for A[M, K] @ weight[N, K].T.  Backend flags are separated from
# kernel meta-parameters by _launch_linear so they are never accidentally
# forwarded as kernel arguments.
LINEAR_CONFIGS = {
    (819200, 1024, 192): {
        "bias": True,
        "family": "memory",
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
        "bias": True,
        "family": "memory",
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
        "bias": True,
        "family": "register",
        "BLOCK_M": 128,
        "BLOCK_N": 128,
        "BLOCK_K": 64,
        "GROUP_M": 1,
        "NUM_XCDS": 1,
        "matrix_instr_nonkdim": 16,
        "waves_per_eu": 0,
        "kpack": 1,
        "num_warps": 4,
        "num_stages": 2,
        "ENABLE_SCHED_BARRIER": True,
        "SINK_INSTS_TO_AVOID_SPILLS": True,
    },
    (2048, 25408, 10240): {
        "bias": False,
        "family": "memory",
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
        "bias": True,
        "family": "memory",
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


def _launch_linear(a, b, bias, out, selected):
    """Launch one already-validated configuration into ``out``."""
    meta = dict(selected)
    family = meta.pop("family")
    meta.pop("bias")

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
        bias.stride(0) if bias is not None else 0,
        bias.stride(1) if bias is not None else 0,
    )
    grid = (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]), )
    kernel = _linear_memory_kernel if family == "memory" else _linear_register_kernel
    kernel[grid](*args, ADD_BIAS=bias is not None, **meta, **backend)


def _validate_linear(a, weight, bias, out):
    if a.ndim != 2 or weight.ndim != 2 or a.shape[1] != weight.shape[1]:
        raise ValueError("Expected A[M, K] and weight[N, K]")
    m, k = a.shape
    n = weight.shape[0]
    shape = (m, n, k)
    selected = LINEAR_CONFIGS.get(shape)
    if selected is None:
        raise ValueError(f"No tuned production gfx942 GEMM configuration for MxNxK={shape}")
    if a.device.type != "cuda" or weight.device != a.device:
        raise ValueError("The tuned gfx942 linear kernel requires A and weight on the same GPU")
    if a.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("The tuned gfx942 linear kernel requires BF16 A and weight")
    if not a.is_contiguous() or not weight.is_contiguous():
        raise ValueError("A[M, K] and weight[N, K] must be contiguous")

    needs_bias = selected["bias"]
    if needs_bias:
        if bias is None or bias.shape != (n, ):
            raise ValueError(f"MxNxK={shape} requires a contiguous bias with shape ({n},)")
        if bias.device != a.device or bias.dtype != a.dtype or not bias.is_contiguous():
            raise ValueError("Bias must be contiguous and match A's device and dtype")
    elif bias is not None:
        raise ValueError(f"MxNxK={shape} is a bias-free shape and does not accept bias")

    if out is not None:
        if out.shape != (m, n) or out.device != a.device or out.dtype != a.dtype or not out.is_contiguous():
            raise ValueError(f"out must be a contiguous BF16 tensor with shape ({m}, {n}) on A's device")
    return selected


def linear(
    a: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a frozen, shape-specialized Linear operation on MI300X.

    ``weight`` uses the common Linear storage layout ``[N, K]``.  The internal
    transpose is a view; it performs no allocation or copy.  Unsupported
    shapes and input combinations raise ``ValueError`` rather than silently dispatching
    to a different provider.
    """
    selected = _validate_linear(a, weight, bias, out)
    m = a.shape[0]
    n = weight.shape[0]
    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    b = weight.t()
    bias_2d = bias.expand(m, n) if bias is not None else None
    _launch_linear(a, b, bias_2d, out, selected)
    return out
