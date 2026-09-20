"""Shared MI300X (gfx942/CDNA3) GEMM implementation for ``mm`` and ``addmm``.

One direct-load kernel serves both operations through a compact heuristic or
full autotune space.
"""

import functools
import importlib
import logging

import torch

import triton
import triton.language as tl

log = logging.getLogger(__name__)

# Origami only models the macro tile, matrix instruction, and occupancy. Keep
# enough analytical groups for Triton's empirical tuner to resolve TLX-specific
# variants that Origami cannot see. This is intentionally opt-in through
# ``space="origami"``; the production heuristic is unchanged.
_ORIGAMI_TOP_K = 8
_ORIGAMI_UNSUPPORTED_ROCM_MAJOR = 10


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
            b = tl.load(b_ptrs)
            a0 = tl.load(a0_ptrs)
            a1 = tl.load(a1_ptrs)
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
        k_main = K if even_k else (K // BLOCK_K) * BLOCK_K
        for k in range(0, k_main, BLOCK_K):
            a_ptrs = a_ptr + reg_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
            b_ptrs = b_ptr + (k + offs_k[:, None]) * stride_bk + reg_n[None, :] * stride_bn
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc, allow_tf32=False, out_dtype=tl.float32)
        if not even_k:
            a_ptrs = a_ptr + reg_m[:, None] * stride_am + (k_main + offs_k[None, :]) * stride_ak
            b_ptrs = b_ptr + (k_main + offs_k[:, None]) * stride_bk + reg_n[None, :] * stride_bn
            tail = offs_k < K - k_main
            a = tl.load(a_ptrs, mask=tail[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=tail[:, None], other=0.0)
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
        "waves_per_eu": waves_per_eu,
        "kpack": kpack,
    }
    if split_m_128_32:
        meta["SPLIT_M_128_32"] = True
    return triton.Config(meta, num_warps=num_warps, num_stages=2)


def _configs():
    """Curated ROCm search space plus every incumbent TLX configuration.

    The first group mirrors PyTorch's ROCm MM candidate pool.  The second keeps
    the previously measured TLX space as a strict subset, so expanding the
    oracle cannot remove a known winner.  Exact duplicates are discarded while
    variants with different warps, grouping, or occupancy remain available to
    the empirical tuner.
    """
    # (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, waves_per_eu)
    candidates = [
        (16, 16, 256, 4, 4, 2),
        (32, 16, 256, 4, 4, 0),
        (32, 32, 16, 8, 4, 2),
        (32, 32, 128, 8, 4, 0),
        (32, 64, 64, 8, 4, 0),
        (64, 16, 128, 8, 4, 2),
        (64, 32, 32, 8, 4, 0),
        (64, 32, 64, 8, 4, 0),
        (64, 32, 64, 8, 8, 0),
        (64, 32, 128, 8, 4, 0),
        (64, 64, 16, 8, 4, 0),
        (64, 64, 64, 4, 4, 0),
        (64, 64, 128, 16, 8, 0),
        (64, 64, 256, 4, 8, 0),
        (64, 128, 32, 4, 4, 2),
        (64, 128, 32, 8, 8, 0),
        (64, 128, 64, 4, 8, 0),
        (64, 128, 128, 4, 8, 0),
        (128, 32, 32, 8, 4, 0),
        (128, 32, 64, 8, 4, 0),
        (128, 64, 32, 8, 4, 2),
        (128, 64, 64, 16, 4, 0),
        (128, 64, 128, 4, 8, 0),
        (128, 128, 32, 16, 4, 2),
        (128, 128, 32, 16, 8, 0),
        (128, 128, 32, 16, 8, 2),
        (128, 128, 64, 16, 4, 0),
        (128, 128, 64, 8, 8, 0),
        (128, 128, 128, 16, 8, 0),
        (128, 256, 32, 16, 4, 2),
        (128, 256, 64, 4, 8, 0),
        (256, 64, 64, 4, 8, 0),
        (256, 128, 32, 4, 4, 2),
        (256, 128, 32, 16, 8, 0),
        (256, 128, 64, 4, 8, 0),
        (256, 256, 64, 4, 8, 0),
        # The original compact TLX space. Some differ from PyTorch only in
        # GROUP_M or warp count, which Origami cannot rank by itself.
        (64, 64, 64, 4, 4, 0),
        (64, 64, 128, 8, 8, 0),
        (128, 64, 64, 4, 8, 0),
        (64, 128, 64, 8, 8, 0),
        (128, 128, 32, 8, 4, 0),
        (128, 128, 64, 8, 8, 0),
        (256, 128, 32, 8, 8, 0),
        (128, 256, 32, 8, 8, 0),
        (256, 256, 64, 8, 8, 0),
    ]
    configs = []
    seen = set()
    for block_m, block_n, block_k, group_m, num_warps, waves_per_eu in candidates:
        key = (block_m, block_n, block_k, group_m, num_warps, waves_per_eu)
        if key not in seen:
            seen.add(key)
            configs.append(_config(block_m, block_n, block_k, group_m, num_warps, waves_per_eu=waves_per_eu))
    return configs


CONFIGS = _configs


def _smoke_configs():
    return [_config(64, 64, 64, 4, 4), _config(128, 128, 32, 8, 4)]


SMOKE_CONFIGS = _smoke_configs


def _load_origami():
    """Load the optional analytical selector when this ROCm version supports it."""
    hip_version = torch.version.hip
    if hip_version is not None:
        try:
            if int(hip_version.split(".", 1)[0]) >= _ORIGAMI_UNSUPPORTED_ROCM_MAJOR:
                return None
        except ValueError:
            return None
    try:
        module = importlib.import_module("origami")
    except (ImportError, OSError):
        return None
    required = (
        "config_t",
        "dim3_t",
        "get_hardware_for_device",
        "problem_t",
        "rank_configs",
        "string_to_datatype",
        "transpose_t",
    )
    return module if all(hasattr(module, name) for name in required) else None


def _origami_transpose(origami, tensor):
    """Translate the executed tensor layout to Origami's row/column-major flag."""
    rows, cols = tensor.shape[-2:]
    stride_rows, stride_cols = tensor.stride()[-2:]
    if cols == 1 or stride_cols == 1:
        return origami.transpose_t.N
    if rows == 1 or stride_rows == 1:
        return origami.transpose_t.T
    raise ValueError(f"Origami does not support matrix strides {tensor.stride()}")


def _origami_config_key(config):
    """Dimensions visible to Origami for one original TLX config."""
    return (
        *(config.kwargs[name] for name in ("BLOCK_M", "BLOCK_N", "BLOCK_K")),
        # waves_per_eu=0 asks the compiler to choose. Origami requires a
        # positive value, so retain the conservative occupancy-one mapping.
        max(1, config.kwargs["waves_per_eu"]),
    )


def _rank_configs_with_origami(origami, configs, named_args, top_k=_ORIGAMI_TOP_K):
    """Rank macro-tile groups and return the corresponding original TLX configs.

    The public low-level API avoids ``OrigamiMatmulSelector``'s private fields
    and, crucially, does not reconstruct TLX configs from an incomplete model.
    ``waves_per_eu=0`` is represented as occupancy one because Origami requires
    a positive value. Explicit occupancy variants remain distinct analytical
    groups. Each selected group expands back to every original TLX config so
    Triton can resolve warps and workgroup mapping empirically.
    """
    a = named_args["a_ptr"]
    b = named_args["b_ptr"]
    M, N, K = (int(named_args[name]) for name in ("M", "N", "K"))
    dtype_name = {torch.float16: "f16", torch.bfloat16: "bf16"}.get(a.dtype)
    if dtype_name is None or b.dtype != a.dtype:
        raise ValueError(f"Origami does not support TLX MM dtypes {a.dtype} and {b.dtype}")

    dtype = origami.string_to_datatype(dtype_name)
    problem = origami.problem_t()
    problem.size = origami.dim3_t(M, N, K)
    problem.batch = 1
    problem.a_transpose = _origami_transpose(origami, a)
    problem.b_transpose = _origami_transpose(origami, b)
    problem.a_dtype = dtype
    problem.b_dtype = dtype
    problem.c_dtype = dtype
    problem.d_dtype = dtype
    problem.mi_dtype = dtype

    device_index = a.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    hardware = origami.get_hardware_for_device(device_index)

    by_model_config = {}
    for config in configs:
        by_model_config.setdefault(_origami_config_key(config), []).append(config)

    analytical = []
    for block_m, block_n, block_k, occupancy in by_model_config:
        config = origami.config_t()
        config.mt = origami.dim3_t(block_m, block_n, block_k)
        # gfx942 FP16/BF16 lowering uses matrix_instr_nonkdim=16.
        config.mi = origami.dim3_t(16, 16, 16)
        config.occupancy = occupancy
        analytical.append(config)

    ranked = origami.rank_configs(problem, hardware, analytical)

    selected = []
    seen = set()
    for result in ranked:
        tile = result.config.mt
        key = (tile.m, tile.n, tile.k, result.config.occupancy)
        if key in seen or key not in by_model_config:
            raise RuntimeError(f"Origami returned an unknown or duplicate config {key}")
        seen.add(key)
        selected.extend(by_model_config[key])
        if len(seen) >= top_k:
            break
    if not selected:
        raise RuntimeError("Origami returned no TLX configs")
    return selected


def _origami_prune_configs(configs, named_args, **kwargs):
    """Select an empirical top-K, falling back to today's one-config heuristic."""
    del kwargs
    origami = _load_origami()
    if origami is not None:
        try:
            return _rank_configs_with_origami(origami, configs, named_args)
        except Exception as exc:
            log.warning("Origami GFX942 MM selection failed; using the existing heuristic: %s", exc)
    M, N, K = (int(named_args[name]) for name in ("M", "N", "K"))
    fallback = heuristic_config(M, N, K)[0]
    for config in configs:
        if (config.kwargs == fallback.kwargs and config.num_warps == fallback.num_warps
                and config.num_stages == fallback.num_stages):
            return [config]
    raise RuntimeError("Origami fallback config is missing from its autotune space")


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


def _candidate_configs(shape):
    """One candidate universe shared by exhaustive and Origami tuning."""
    configs = CONFIGS()
    incumbent = heuristic_config(*shape)[0]
    incumbent_key = (incumbent.kwargs, incumbent.num_warps, incumbent.num_stages)
    if not any((config.kwargs, config.num_warps, config.num_stages) == incumbent_key for config in configs):
        configs.append(incumbent)
    return configs


@functools.lru_cache(maxsize=None)
def _tuned(space, shape=None):
    """Autotuned direct-load kernel per search space."""
    if space == "heuristic":
        configs = heuristic_config(*shape)
    elif space in ("full", "origami"):
        configs = _candidate_configs(shape)
    elif space == "smoke":
        configs = SMOKE_CONFIGS()
    else:
        raise ValueError(f"Unknown gfx942 MM search space: {space}")
    keys = ["M", "N", "K", "ADD_BIAS"]
    prune_configs_by = None
    if space in ("full", "origami"):
        # Origami models operand orientation, so layout must participate in the
        # autotune cache key even when two calls share M/N/K and dtype. Full
        # tuning uses the same key so it remains a valid oracle.
        keys += ["stride_am", "stride_ak", "stride_bk", "stride_bn"]
    if space == "origami":
        prune_configs_by = {"early_config_prune": _origami_prune_configs}
    return triton.autotune(configs=configs, key=keys, prune_configs_by=prune_configs_by)(matmul_kernel_gfx942)


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

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )  # noqa: E731
    kernel = _tuned(space, (M, N, K) if space in ("heuristic", "origami", "full") else None)
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
        matrix_instr_nonkdim=16,
    )
    return out


def mm(a, b, *, out=None, space="heuristic"):
    """Compute ``a @ b`` using the gfx942 direct-load GEMM kernel."""
    return _gemm(a, b, out=out, space=space)


# Compatibility entry point used by the kernel-optimization agent.
matmul = mm
