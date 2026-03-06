# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
Test to verify that the flash attention kernel uses the expected layout.

This test compiles flash attention kernels (forward, backward preprocess,
and backward main) mirroring the computation patterns from
third-party/triton/beta/triton/python/tutorials/06-fused-attention.py,
retrieves the generated ttgir, and verifies that the blocked layouts match
the expected patterns regardless of compiler upgrades.

The expected layout is determined by the Triton compiler's Coalesce pass
which optimizes memory access patterns. For contiguous loads of fp16 data,
the Coalesce pass sets sizePerThread along the contiguous dimension to
min(128/elemBits, max(numElems/numThreads, 1)), then BlockedEncodingAttr::get
distributes threads and warps across dimensions.
"""

import re

import pytest
import torch
import triton
import triton.language as tl


def parse_layout_params(layout_str: str) -> dict | None:
    """
    Parse a blocked layout string and extract its parameters.

    Args:
        layout_str: A layout string like
            "#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], ...}>"

    Returns:
        A dict with extracted parameters, or None if no parameters found.
    """
    params = {}

    match = re.search(r"sizePerThread\s*=\s*\[([^\]]+)\]", layout_str)
    if match:
        params["sizePerThread"] = [int(x.strip()) for x in match.group(1).split(",")]

    match = re.search(r"threadsPerWarp\s*=\s*\[([^\]]+)\]", layout_str)
    if match:
        params["threadsPerWarp"] = [int(x.strip()) for x in match.group(1).split(",")]

    match = re.search(r"warpsPerCTA\s*=\s*\[([^\]]+)\]", layout_str)
    if match:
        params["warpsPerCTA"] = [int(x.strip()) for x in match.group(1).split(",")]

    match = re.search(r"order\s*=\s*\[([^\]]+)\]", layout_str)
    if match:
        params["order"] = [int(x.strip()) for x in match.group(1).split(",")]

    return params if params else None


def extract_all_blocked_layouts(ttgir_content: str) -> list[tuple[str, dict]]:
    """
    Extract all blocked layout definitions from ttgir content.

    Returns a list of (name, params) tuples, e.g.:
        [("#blocked", {...}), ("#blocked1", {...}), ...]
    """
    pattern = r"(#blocked\d*)\s*=\s*(#ttg\.blocked<\{[^}]+\}>)"
    layouts = []
    for match in re.finditer(pattern, ttgir_content):
        name = match.group(1)
        layout_str = match.group(2)
        params = parse_layout_params(layout_str)
        if params:
            layouts.append((name, params))
    return layouts


def parse_slice_layout(layout_str: str) -> dict | None:
    """
    Parse a slice layout string and extract its parameters.

    Args:
        layout_str: A layout string like "#ttg.slice<{dim = 1, parent = #blocked}>"

    Returns:
        A dict with 'dim' and 'parent' keys, or None if parsing fails.
    """
    params = {}

    dim_match = re.search(r"dim\s*=\s*(\d+)", layout_str)
    if dim_match:
        params["dim"] = int(dim_match.group(1))

    parent_match = re.search(r"parent\s*=\s*(#\w+)", layout_str)
    if parent_match:
        params["parent"] = parent_match.group(1)

    return params if params else None


def extract_reduce_output_layouts(ttgir_content: str) -> list[dict]:
    """
    Extract the output layouts from all tt.reduce operations in ttgir content.

    Returns:
        A list of dicts with 'dim' and 'parent' keys describing the slice layouts.
    """
    reduce_pattern = (
        r'"tt\.reduce"'
        r"[\s\S]*?"
        r"\}\)\s*:\s*"
        r"\([^)]+\)\s*->\s*"
        r"tensor<[^,]+,\s*(#ttg\.slice<\{[^}]+\}>)>"
    )

    results = []
    for match in re.finditer(reduce_pattern, ttgir_content):
        slice_layout = match.group(1)
        params = parse_slice_layout(slice_layout)
        if params:
            results.append(params)

    return results


def check_layout_matches(actual_params: dict, expected_params: dict) -> tuple[bool, str]:
    """
    Check if actual layout parameters match expected parameters.

    Returns:
        (matches, message) tuple
    """
    mismatches = []
    for key, expected_value in expected_params.items():
        if key not in actual_params:
            mismatches.append(f"  {key}: expected {expected_value}, but key not found")
        elif actual_params[key] != expected_value:
            mismatches.append(
                f"  {key}: expected {expected_value}, got {actual_params[key]}"
            )

    if mismatches:
        return False, (
            f"Layout mismatch:\n"
            + "\n".join(mismatches)
            + f"\nActual params: {actual_params}"
        )

    return True, f"Layout matches: {actual_params}"


def get_warp_size() -> int:
    """
    Get the warp size for the current GPU.

    Returns:
        Warp size: 64 for AMD GPUs (wavefront), 32 for NVIDIA GPUs

    Raises:
        RuntimeError: If CUDA/ROCm is not available or GPU type is not recognized
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm not available")

    device_name = torch.cuda.get_device_name(0).lower()

    if (
        "amd" in device_name
        or "radeon" in device_name
        or "mi" in device_name
        or "gfx" in device_name
    ):
        return 64

    if (
        "nvidia" in device_name
        or "geforce" in device_name
        or "tesla" in device_name
        or "quadro" in device_name
        or "rtx" in device_name
        or "gtx" in device_name
        or "a100" in device_name
        or "h100" in device_name
        or "h200" in device_name
        or "b100" in device_name
        or "b200" in device_name
        or "v100" in device_name
        or "p100" in device_name
    ):
        return 32

    raise RuntimeError(
        f"Unsupported GPU type: '{device_name}'. "
        "Expected AMD GPU (warp size 64) or NVIDIA GPU (warp size 32). "
        "Please update get_warp_size() to support this GPU."
    )


# ---------------------------------------------------------------------------
# Flash attention forward kernel (simplified, non-autotuned version for
# layout testing).  This mirrors the core computation of _attn_fwd from
# 06-fused-attention.py but avoids autotuning and tensor-descriptor
# complexity so that we can directly inspect the compiled ttgir.
# ---------------------------------------------------------------------------


@triton.jit
def _flash_attn_fwd_layout_test(
    Q,
    K,
    V,
    Out,
    sm_scale,
    N_CTX,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
):
    """
    Simplified flash attention forward kernel for layout testing.

    This kernel captures the core computation pattern of the flash attention
    forward pass: Q*K^T dot product, softmax-like reduction, and P*V dot
    product. It uses pointer-based loads (not tensor descriptors) for
    simplicity.
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    # Load Q tile: [BLOCK_M, HEAD_DIM]
    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504  # 1/log(2)

    # Determine loop bounds based on STAGE
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX

    # Loop over K, V blocks
    for start_n in range(lo, hi, BLOCK_N):
        # Load K tile: [BLOCK_N, HEAD_DIM]
        k_ptrs = K + k_offset + (start_n + offs_n)[:, None] * stride_kn + offs_k[None, :] * stride_kk
        k = tl.load(k_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)

        # Compute QK^T: [BLOCK_M, BLOCK_N] = [BLOCK_M, HEAD_DIM] x [HEAD_DIM, BLOCK_N]
        qk = tl.dot(q, tl.trans(k))

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)

        acc = acc * alpha[:, None]

        # Load V tile: [BLOCK_N, HEAD_DIM]
        v_ptrs = V + v_offset + (start_n + offs_n)[:, None] * stride_vn + offs_k[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=(start_n + offs_n)[:, None] < N_CTX, other=0.0)

        # Compute P*V: [BLOCK_M, HEAD_DIM] = [BLOCK_M, BLOCK_N] x [BLOCK_N, HEAD_DIM]
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # Normalize output
    acc = acc / l_i[:, None]

    # Store output: [BLOCK_M, HEAD_DIM]
    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < N_CTX)


# ---------------------------------------------------------------------------
# Backward preprocess kernel (from 06-fused-attention.py _attn_bwd_preprocess)
# ---------------------------------------------------------------------------


@triton.jit
def _flash_attn_bwd_preprocess_layout_test(
    O,
    DO,
    Delta,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Backward preprocess: computes delta = sum(o * do, axis=1)."""
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# ---------------------------------------------------------------------------
# Backward main kernel (from 06-fused-attention.py _attn_bwd, _attn_bwd_dkdv,
# _attn_bwd_dq). This mirrors the full backward pass computation patterns
# including multiple dot products across different operand shapes.
# ---------------------------------------------------------------------------


@triton.jit
def _attn_bwd_dkdv_layout_test(
    dk, dv,
    Q, k, v, sm_scale,
    DO,
    M, D,
    stride_tok, stride_d,
    H, N_CTX,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    start_n, start_m, num_steps,
    MASK: tl.constexpr,
):
    """Compute dK and dV for a block of K/V rows."""
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        # [BLOCK_N1, HEAD_DIM] x [HEAD_DIM, BLOCK_M1] -> [BLOCK_N1, BLOCK_M1]
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        if MASK:
            mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs)
        # [BLOCK_N1, BLOCK_M1] x [BLOCK_M1, HEAD_DIM] -> [BLOCK_N1, HEAD_DIM]
        ppT = pT.to(tl.float16)
        dv += tl.dot(ppT, do)
        Di = tl.load(D + offs_m)
        # [HEAD_DIM, BLOCK_N1]^T x [BLOCK_M1, HEAD_DIM]^T -> [BLOCK_N1, BLOCK_M1]
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        # [BLOCK_N1, BLOCK_M1] x [BLOCK_M1, HEAD_DIM] -> [BLOCK_N1, HEAD_DIM]
        dk += tl.dot(dsT, tl.trans(qT))
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
    return dk, dv


@triton.jit
def _attn_bwd_dq_layout_test(
    dq, q, K, V,
    do, m, D,
    stride_tok, stride_d,
    H, N_CTX,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    start_m, start_n, num_steps,
    MASK: tl.constexpr,
):
    """Compute dQ for a block of Q rows."""
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    Di = tl.load(D + offs_m)
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        # [BLOCK_M2, HEAD_DIM] x [HEAD_DIM, BLOCK_N2] -> [BLOCK_M2, BLOCK_N2]
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = offs_m[:, None] >= offs_n[None, :]
            p = tl.where(mask, p, 0.0)
        # [BLOCK_M2, HEAD_DIM] x [HEAD_DIM, BLOCK_N2] -> [BLOCK_M2, BLOCK_N2]
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        # [BLOCK_M2, BLOCK_N2] x [BLOCK_N2, HEAD_DIM] -> [BLOCK_M2, HEAD_DIM]
        dq += tl.dot(ds, tl.trans(kT))
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


@triton.jit
def _flash_attn_bwd_layout_test(
    Q, K, V, sm_scale,
    DO, DQ, DK, DV,
    M, D,
    stride_z, stride_h, stride_tok, stride_d,
    H, N_CTX,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """
    Simplified flash attention backward kernel for layout testing.

    This mirrors _attn_bwd from 06-fused-attention.py. It computes dK, dV
    (via _attn_bwd_dkdv) and dQ (via _attn_bwd_dq) using pointer-based loads.
    The key computation patterns are:
    - dkdv: k @ qT, ppT @ do, v @ do^T, dsT @ qT^T
    - dq: q @ kT, do @ vT, ds @ kT^T
    """
    LN2: tl.constexpr = 0.6931471824645996

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = 0

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # Load K and V: [BLOCK_N1, HEAD_DIM]
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    if CAUSAL:
        start_m = start_n
        num_steps = BLOCK_N1 // MASK_BLOCK_M1
        dk, dv = _attn_bwd_dkdv_layout_test(
            dk, dv,
            Q, k, v, sm_scale, DO, M, D,
            stride_tok, stride_d, H, N_CTX,
            MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,
            start_n, start_m, num_steps,
            MASK=True,
        )
        start_m += num_steps * MASK_BLOCK_M1

    num_steps = (N_CTX - start_m) // BLOCK_M1
    dk, dv = _attn_bwd_dkdv_layout_test(
        dk, dv,
        Q, k, v, sm_scale, DO, M, D,
        stride_tok, stride_d, H, N_CTX,
        BLOCK_M1, BLOCK_N1, HEAD_DIM,
        start_n, start_m, num_steps,
        MASK=False,
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # DQ computation
    start_m = pid * BLOCK_M2
    start_n = 0
    num_steps = N_CTX // BLOCK_N2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    if CAUSAL:
        end_n = start_m + BLOCK_M2
        num_steps = BLOCK_M2 // MASK_BLOCK_N2
        dq = _attn_bwd_dq_layout_test(
            dq, q, K, V, do, m, D,
            stride_tok, stride_d, H, N_CTX,
            BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,
            start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,
            MASK=True,
        )
        end_n -= num_steps * MASK_BLOCK_N2
        num_steps = end_n // BLOCK_N2
        start_n = end_n - num_steps * BLOCK_N2

    dq = _attn_bwd_dq_layout_test(
        dq, q, K, V, do, m, D,
        stride_tok, stride_d, H, N_CTX,
        BLOCK_M2, BLOCK_N2, HEAD_DIM,
        start_m, start_n, num_steps,
        MASK=False,
    )

    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


# ---------------------------------------------------------------------------
# Expected layout calculation
# ---------------------------------------------------------------------------


def _compute_blocked_encoding(
    shape: list[int],
    size_per_thread: list[int],
    order: list[int],
    num_warps: int,
    threads_per_warp: int,
) -> dict:
    """
    Compute the expected BlockedEncodingAttr parameters.

    This mirrors the BlockedEncodingAttr::get builder logic in
    TritonGPUAttrDefs.td (lines 946-982). Starting from the contiguous
    dimension, it distributes threads across dimensions based on the shape
    and sizePerThread.

    Args:
        shape: Tensor shape (e.g., [128, 128])
        size_per_thread: Elements per thread per dimension (e.g., [1, 8])
        order: Dimension ordering, contiguous first (e.g., [1, 0])
        num_warps: Number of warps per CTA
        threads_per_warp: Threads per warp (warp size)

    Returns:
        Dict with sizePerThread, threadsPerWarp, warpsPerCTA, order
    """
    rank = len(shape)
    tpw = [0] * rank
    wpc = [0] * rank

    remaining_lanes = threads_per_warp
    remaining_threads = num_warps * threads_per_warp
    remaining_warps = num_warps
    prev_lanes = 1
    prev_warps = 1

    # Starting from the contiguous dimension
    for d in range(rank - 1):
        i = order[d]
        threads_per_cta = min(
            max(remaining_threads, 1),
            max(1, shape[i] // size_per_thread[i]),
        )
        tpw[i] = min(max(threads_per_cta, 1), remaining_lanes)
        wpc[i] = min(max(threads_per_cta // tpw[i], 1), remaining_warps)
        remaining_warps //= wpc[i]
        remaining_lanes //= tpw[i]
        remaining_threads //= threads_per_cta
        prev_lanes *= tpw[i]
        prev_warps *= wpc[i]

    # Expand the last dimension to fill remaining lanes and warps
    tpw[order[rank - 1]] = threads_per_warp // prev_lanes
    wpc[order[rank - 1]] = num_warps // prev_warps

    return {
        "sizePerThread": size_per_thread,
        "threadsPerWarp": tpw,
        "warpsPerCTA": wpc,
        "order": order,
    }


def get_expected_coalesced_params(
    shape: list[int],
    num_warps: int,
    warp_size: int,
    elem_bits: int = 16,
) -> dict:
    """
    Calculate expected blocked layout after the Coalesce pass.

    The Coalesce pass (Coalesce.cpp) optimizes memory access patterns for
    loads/stores. For contiguous fp16 loads:

    1. Compute perThread = min(128/elemBits, max(numElems/numThreads, 1))
       - 128 bits is the maximum vectorized load width
       - elemBits is typically 16 for fp16
       - perThread is capped at 8 for fp16 (128/16 = 8)

    2. Set sizePerThread[contiguous_dim] = perThread

    3. BlockedEncodingAttr::get then distributes threads and warps based
       on the shape and sizePerThread (TritonGPUAttrDefs.td lines 946-982).

    Args:
        shape: 2D tensor shape (e.g., [128, 128])
        num_warps: Number of warps per CTA
        warp_size: Number of threads per warp (64 for AMD, 32 for NVIDIA)
        elem_bits: Bits per element (default 16 for fp16)

    Returns:
        Dictionary with expected layout parameters
    """
    num_elems = 1
    for s in shape:
        num_elems *= s
    num_threads = num_warps * warp_size

    # Coalesce pass: compute perThread for contiguous loads
    max_per_thread = 128 // elem_bits  # max vectorized load width
    per_thread = min(max_per_thread, max(num_elems // num_threads, 1))

    # order=[1, 0]: contiguous dimension is 1 (last dim / feature dim)
    order = [1, 0]
    size_per_thread = [1, per_thread]

    return _compute_blocked_encoding(
        shape, size_per_thread, order, num_warps, warp_size
    )


def find_layout_by_params_subset(
    layouts: list[tuple[str, dict]],
    expected: dict,
) -> tuple[str, dict] | None:
    """
    Find a layout whose parameters match a subset of expected parameters.

    Returns the first (name, params) tuple where all keys in expected
    match, or None if no match found.
    """
    for name, params in layouts:
        matches = True
        for key, value in expected.items():
            if key not in params or params[key] != value:
                matches = False
                break
        if matches:
            return (name, params)
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.parametrize("num_warps", [4, 8])
def test_flash_attn_fwd_layout(HEAD_DIM, num_warps):
    """
    Test that the flash attention forward kernel uses the expected blocked layout.

    This test compiles the flash attention forward kernel, retrieves the
    generated ttgir, and verifies that the blocked layout for the main
    computation (Q/K/V loads and stores) matches the expected pattern
    determined by the compiler's Coalesce pass.

    Uses the same kernel launch parameter configs from
    06-fused-attention.py (pytest config: BLOCK_M=128, BLOCK_N=64).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm not available")

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float16

    # Fixed block sizes matching the tutorial's pytest config
    BLOCK_M = 128
    BLOCK_N = 64
    N_CTX = 256
    Z = 1
    H = 1

    # Create input tensors
    q = torch.randn(Z, H, N_CTX, HEAD_DIM, device=device, dtype=dtype)
    k = torch.randn(Z, H, N_CTX, HEAD_DIM, device=device, dtype=dtype)
    v = torch.randn(Z, H, N_CTX, HEAD_DIM, device=device, dtype=dtype)
    o = torch.empty_like(q)

    sm_scale = 0.5
    STAGE = 1  # non-causal

    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H)

    compiled_kernel = _flash_attn_fwd_layout_test[grid](
        q, k, v, o,
        sm_scale, N_CTX,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Z, H,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        STAGE=STAGE,
        num_warps=num_warps,
    )

    # Get the ttgir
    ttgir = compiled_kernel.asm["ttgir"]

    # Extract all blocked layouts from ttgir
    layouts = extract_all_blocked_layouts(ttgir)
    assert len(layouts) > 0, (
        f"No blocked layouts found in ttgir.\nttgir content:\n{ttgir[:2000]}"
    )

    warp_size = get_warp_size()

    # The primary blocked layout corresponds to the tensor shape used for
    # loads/stores: [BLOCK_M, HEAD_DIM] for Q and output, [BLOCK_N, HEAD_DIM]
    # for K and V. The Coalesce pass determines sizePerThread based on
    # memory access contiguity and element bit width (fp16 = 16 bits).
    # Both [BLOCK_M, HEAD_DIM] and [BLOCK_N, HEAD_DIM] loads produce the
    # same coalesced layout since they share the same HEAD_DIM contiguous axis.
    expected_primary = get_expected_coalesced_params(
        [BLOCK_M, HEAD_DIM], num_warps, warp_size, elem_bits=16
    )

    found = find_layout_by_params_subset(layouts, expected_primary)
    assert found is not None, (
        f"No blocked layout matching expected primary layout found.\n"
        f"Expected: {expected_primary}\n"
        f"Found layouts: {layouts}\n"
        f"BLOCK_M={BLOCK_M}, HEAD_DIM={HEAD_DIM}, num_warps={num_warps}, "
        f"warp_size={warp_size}"
    )

    _, actual_params = found
    matches, message = check_layout_matches(actual_params, expected_primary)
    assert matches, (
        f"Primary blocked layout does not match expected pattern.\n{message}"
    )

    # Verify reduce output layouts (from tl.max and tl.sum along axis=1)
    # These should produce slice layouts with dim=1.
    # The parent layout type varies by GPU architecture: #blocked on older
    # GPUs, #linear on Blackwell (MMAv5 uses linear/tensor-memory layouts
    # for dot results). We only check that the reduce dimension is correct.
    reduce_layouts = extract_reduce_output_layouts(ttgir)
    for i, reduce_layout in enumerate(reduce_layouts):
        assert reduce_layout.get("dim") == 1, (
            f"Reduce output layout {i} has unexpected dim.\n"
            f"Expected dim=1, got dim={reduce_layout.get('dim')}\n"
            f"Full layout: {reduce_layout}"
        )


@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.parametrize("num_warps", [4, 8])
def test_flash_attn_bwd_preprocess_layout(HEAD_DIM, num_warps):
    """
    Test that the flash attention backward preprocess kernel uses the expected layout.

    The backward preprocess kernel computes delta = sum(o * do, axis=1),
    operating on [BLOCK_M, HEAD_DIM] shaped tensors.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm not available")

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float16

    BLOCK_M = 128
    N_CTX = 256
    Z = 1
    H = 1

    o = torch.randn(Z * H, N_CTX, HEAD_DIM, device=device, dtype=dtype)
    do = torch.randn_like(o)
    delta = torch.empty(Z * H, N_CTX, device=device, dtype=torch.float32)

    pre_grid = (N_CTX // BLOCK_M, Z * H)

    compiled_kernel = _flash_attn_bwd_preprocess_layout_test[pre_grid](
        o, do, delta,
        Z, H, N_CTX,
        BLOCK_M=BLOCK_M,
        HEAD_DIM=HEAD_DIM,
        num_warps=num_warps,
    )

    ttgir = compiled_kernel.asm["ttgir"]

    layouts = extract_all_blocked_layouts(ttgir)
    assert len(layouts) > 0, (
        f"No blocked layouts found in ttgir.\nttgir content:\n{ttgir[:2000]}"
    )

    warp_size = get_warp_size()

    # The blocked layout corresponds to [BLOCK_M, HEAD_DIM] loads of fp16 data
    expected = get_expected_coalesced_params(
        [BLOCK_M, HEAD_DIM], num_warps, warp_size, elem_bits=16
    )

    found = find_layout_by_params_subset(layouts, expected)
    assert found is not None, (
        f"No blocked layout matching expected layout found.\n"
        f"Expected: {expected}\n"
        f"Found layouts: {layouts}\n"
        f"BLOCK_M={BLOCK_M}, HEAD_DIM={HEAD_DIM}, num_warps={num_warps}, "
        f"warp_size={warp_size}"
    )

    _, actual_params = found
    matches, message = check_layout_matches(actual_params, expected)
    assert matches, (
        f"Backward preprocess blocked layout does not match expected pattern.\n"
        f"{message}"
    )

    # Verify the reduce output layout (sum along axis=1).
    # The parent layout type is typically #blocked for non-dot operations,
    # but may vary by architecture. We check dim=1 and accept known parents.
    reduce_layouts = extract_reduce_output_layouts(ttgir)
    valid_parents = {"#blocked", "#linear"}
    for i, reduce_layout in enumerate(reduce_layouts):
        assert reduce_layout.get("dim") == 1, (
            f"Reduce output layout {i} has unexpected dim.\n"
            f"Expected dim=1, got dim={reduce_layout.get('dim')}\n"
            f"Full layout: {reduce_layout}"
        )
        parent = reduce_layout.get("parent")
        assert parent in valid_parents, (
            f"Reduce output layout {i} has unexpected parent.\n"
            f"Expected one of {valid_parents}, got {parent}\n"
            f"Full layout: {reduce_layout}"
        )


@pytest.mark.parametrize("HEAD_DIM", [64, 128])
@pytest.mark.parametrize("num_warps", [4])
def test_flash_attn_bwd_layout(HEAD_DIM, num_warps):
    """
    Test that the flash attention backward kernel uses the expected blocked layout.

    The backward kernel (_attn_bwd) contains multiple dot products across
    different operand shapes:
    - dkdv path: k @ qT [BLOCK_N1, HEAD_DIM] x [HEAD_DIM, BLOCK_M1],
                 ppT @ do [BLOCK_N1, BLOCK_M1] x [BLOCK_M1, HEAD_DIM],
                 v @ do^T [BLOCK_N1, HEAD_DIM] x [HEAD_DIM, BLOCK_M1],
                 dsT @ qT^T [BLOCK_N1, BLOCK_M1] x [BLOCK_M1, HEAD_DIM]
    - dq path:   q @ kT [BLOCK_M2, HEAD_DIM] x [HEAD_DIM, BLOCK_N2],
                 do @ vT [BLOCK_M2, HEAD_DIM] x [HEAD_DIM, BLOCK_N2],
                 ds @ kT^T [BLOCK_M2, BLOCK_N2] x [BLOCK_N2, HEAD_DIM]

    Uses the same block sizes as the tutorial's backward pass:
    BLOCK_M1=32, BLOCK_N1=128, BLOCK_M2=128, BLOCK_N2=32, BLK_SLICE_FACTOR=2.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm not available")

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float16

    # Block sizes from the tutorial's backward pass (line 595)
    BLOCK_M1 = 32
    BLOCK_N1 = 128
    BLOCK_M2 = 128
    BLOCK_N2 = 32
    BLK_SLICE_FACTOR = 2
    N_CTX = 256
    Z = 1
    H = 1
    CAUSAL = False

    # Create input tensors matching the backward pass shapes
    q = torch.randn(Z, H, N_CTX, HEAD_DIM, device=device, dtype=dtype)
    k = torch.randn(Z, H, N_CTX, HEAD_DIM, device=device, dtype=dtype)
    v = torch.randn(Z, H, N_CTX, HEAD_DIM, device=device, dtype=dtype)
    do = torch.randn(Z, H, N_CTX, HEAD_DIM, device=device, dtype=dtype)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    # Pre-scale k as done in the tutorial (line 599)
    RCP_LN2 = 1.4426950408889634
    sm_scale = 0.5
    k_scaled = k * (sm_scale * RCP_LN2)

    # M (logsumexp) and Delta from forward pass
    M_tensor = torch.randn(Z * H, N_CTX, device=device, dtype=torch.float32)
    delta = torch.randn(Z * H, N_CTX, device=device, dtype=torch.float32)

    grid = (N_CTX // BLOCK_N1, 1, Z * H)

    compiled_kernel = _flash_attn_bwd_layout_test[grid](
        q, k_scaled, v, sm_scale,
        do, dq, dk, dv,
        M_tensor, delta,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        H, N_CTX,
        BLOCK_M1=BLOCK_M1,
        BLOCK_N1=BLOCK_N1,
        BLOCK_M2=BLOCK_M2,
        BLOCK_N2=BLOCK_N2,
        BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
        HEAD_DIM=HEAD_DIM,
        CAUSAL=CAUSAL,
        num_warps=num_warps,
        num_stages=5,
    )

    ttgir = compiled_kernel.asm["ttgir"]

    layouts = extract_all_blocked_layouts(ttgir)
    assert len(layouts) > 0, (
        f"No blocked layouts found in ttgir.\nttgir content:\n{ttgir[:2000]}"
    )

    warp_size = get_warp_size()

    # The backward kernel has loads/stores for multiple tensor shapes:
    # - [BLOCK_N1, HEAD_DIM] = [128, HEAD_DIM] for K, V, dK, dV
    # - [BLOCK_M1, HEAD_DIM] = [32, HEAD_DIM] for Q (transposed access), DO
    # - [BLOCK_M2, HEAD_DIM] = [128, HEAD_DIM] for Q, DO, dQ
    # - [HEAD_DIM, BLOCK_M1] = [HEAD_DIM, 32] for qT loads
    # - [HEAD_DIM, BLOCK_N2] = [HEAD_DIM, 32] for kT, vT loads
    # Check that at least the primary load shapes produce matching coalesced
    # layouts. The [BLOCK_N1, HEAD_DIM] and [BLOCK_M2, HEAD_DIM] loads both
    # have shape [128, HEAD_DIM] and should produce the same layout.
    expected_128 = get_expected_coalesced_params(
        [128, HEAD_DIM], num_warps, warp_size, elem_bits=16
    )

    found_128 = find_layout_by_params_subset(layouts, expected_128)
    assert found_128 is not None, (
        f"No blocked layout matching expected [128, HEAD_DIM] layout found.\n"
        f"Expected: {expected_128}\n"
        f"Found layouts: {layouts}\n"
        f"HEAD_DIM={HEAD_DIM}, num_warps={num_warps}, warp_size={warp_size}"
    )

    _, actual_128 = found_128
    matches, message = check_layout_matches(actual_128, expected_128)
    assert matches, (
        f"[128, HEAD_DIM] blocked layout does not match expected pattern.\n"
        f"{message}"
    )

    # Also check the [32, HEAD_DIM] shaped loads (BLOCK_M1 or BLOCK_N2)
    expected_32 = get_expected_coalesced_params(
        [32, HEAD_DIM], num_warps, warp_size, elem_bits=16
    )

    found_32 = find_layout_by_params_subset(layouts, expected_32)
    assert found_32 is not None, (
        f"No blocked layout matching expected [32, HEAD_DIM] layout found.\n"
        f"Expected: {expected_32}\n"
        f"Found layouts: {layouts}\n"
        f"HEAD_DIM={HEAD_DIM}, num_warps={num_warps}, warp_size={warp_size}"
    )

    _, actual_32 = found_32
    matches, message = check_layout_matches(actual_32, expected_32)
    assert matches, (
        f"[32, HEAD_DIM] blocked layout does not match expected pattern.\n"
        f"{message}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
