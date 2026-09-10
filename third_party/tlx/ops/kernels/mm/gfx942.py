"""MI300X (gfx942 / CDNA3) GEMM -- the `tlx.ops.mm` implementation.
"""
import functools
import os

import torch
import triton
import triton.language as tl

from ._shapes import GFX942_FOCUS

#: The shapes `bench_mm.py` gates on for this arch. Correctness runs the union
#: of every arch's list; perf runs only its own.
PERF_SHAPES = GFX942_FOCUS

# MI300X: 8 XCDs, 304 CUs. Consecutive program ids are dispatched round-robin
# across the XCDs, so the remap below undoes that to restore tile locality.
NUM_XCDS = 8

#: Tiles each XCD takes at a time in `_xcd_chunk_remap`. Swept over 1/4/8/16 on
#: the focus shapes: 1 (which degenerates to no remap at all) is 3-13% down,
#: and 8 and 16 trade places with 4 shape by shape without beating it overall.
XCD_CHUNK = 4

# Per-workgroup LDS on CDNA3. Configs are checked against this so an oversized
# tile is dropped before compilation instead of failing out-of-resources.
CDNA3_LDS_BYTES = 64 * 1024


@triton.jit
def _xcd_remap(pid, grid_mn, num_xcds: tl.constexpr):
    """Undo the hardware's round-robin XCD dispatch of consecutive program ids.
    """
    pids_per_xcd = (grid_mn + num_xcds - 1) // num_xcds
    tall_xcds = grid_mn % num_xcds
    tall_xcds = num_xcds if tall_xcds == 0 else tall_xcds
    xcd = pid % num_xcds
    local_pid = pid // num_xcds
    if xcd < tall_xcds:
        return xcd * pids_per_xcd + local_pid
    return tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid


@triton.jit
def _xcd_chunk_remap(pid, grid_mn, num_xcds: tl.constexpr, chunk: tl.constexpr):
    """`_xcd_remap`, but striping `chunk` tiles at a time instead of one slice each.
    """
    aligned = (grid_mn // (num_xcds * chunk)) * (num_xcds * chunk)
    if pid < aligned:
        xcd = pid % num_xcds
        local_pid = pid // num_xcds
        return (local_pid // chunk) * num_xcds * chunk + xcd * chunk + local_pid % chunk
    return pid


@triton.jit
def matmul_kernel_gfx942(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    XCD_CHUNK: tl.constexpr,
    EVEN_K: tl.constexpr,
    PEEL_K_TAIL: tl.constexpr,
    ALIGN_ROWS: tl.constexpr,
):
    """C = A @ B, register-staged; the compiler pipelines global -> LDS -> MFMA."""
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    pid = tl.program_id(0).to(tl.int32)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    if NUM_XCDS != 1:
        pid = _xcd_chunk_remap(pid, num_pid_m * num_pid_n, NUM_XCDS, XCD_CHUNK)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int32)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)) % N
    offs_k = tl.arange(0, BLOCK_K).to(tl.int32)
    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if PEEL_K_TAIL:
        # Apply mask on tail K tile only
        k_main = K if EVEN_K else (K // BLOCK_K) * BLOCK_K
        for k in range(0, k_main, BLOCK_K):
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
            b_ptrs = b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
            acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc, out_dtype=tl.float32)

        if not EVEN_K:
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k_main + offs_k[None, :]) * stride_ak
            b_ptrs = b_ptr + (k_main + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
            tail = offs_k < K - k_main
            a = tl.load(a_ptrs, mask=tail[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=tail[:, None], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    else:
        # This else-branch is just for illustration purposes and should NEVER be invoked
        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
            k = k_idx * BLOCK_K
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
            b_ptrs = b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            else:
                tail = offs_k < K - k
                a = tl.load(a_ptrs, mask=tail[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=tail[:, None], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    c = acc.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int32)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int32)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def lds_bytes(block_m, block_n, block_k, num_buffers, elem_bytes=2):
    """LDS footprint of a tile, ignoring shared-layout padding."""
    return (block_m * block_k + block_k * block_n) * elem_bytes * num_buffers


"""
Ask compiler pipeliner to overlap loading the next
K tile with MFMA on the current tile
"""
_NUM_STAGES = 2


def _config(block_m, block_n, block_k, group_m, num_warps):
    return triton.Config(
        {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
            "GROUP_M": group_m,
            "NUM_XCDS": NUM_XCDS,
            "XCD_CHUNK": XCD_CHUNK,
            "waves_per_eu": 0,
        },
        num_warps=num_warps,
        num_stages=_NUM_STAGES,
    )


def _configs():
    """Tiles worth trying on MI300X. Used by `space="full"`.
    256x256x64 is the largest block configs to fit in LDS.
    """
    tiles = [
        (64, 64, 64, 4),
        (128, 128, 32, 4),
        (128, 128, 64, 4),
        (128, 128, 64, 8),
        (256, 128, 64, 8),
        (128, 256, 64, 8),
        (256, 256, 32, 8),
        (256, 256, 64, 8),
    ]
    return [_config(bm, bn, bk, gm, warps) for (bm, bn, bk, warps) in tiles for gm in (1, 4, 8, 16)]


CONFIGS = _configs


def _smoke_configs():
    """One config per distinct lowering path, for shapes the heuristic declines.

    What differs here is the tile width (which selects the LDS depth the
    pipeliner can afford), the two warp counts, and the swizzle degenerating at
    GROUP_M=1.
    """
    return [
        _config(64, 64, 64, 1, 4),
        _config(128, 128, 32, 8, 4),
        _config(256, 256, 64, 8, 8),
    ]


SMOKE_CONFIGS = _smoke_configs


def heuristic_config(M, N, K):
    """
    TODO. replace with decision tree
    """
    shape = (M, N, K)
    if shape in [
        (819200, 192, 1024),
        (61440, 5120, 2048),
        (2252800, 256, 256),
        (4096, 4096, 2048),
        (61440, 5120, 7744),
    ]:
        return [_config(256, 256, 64, 8, 8)]
    elif shape in [
        (4096, 242432, 1894),
        (61440, 3840, 4096),
    ]:
        return [_config(256, 256, 64, 16, 8)]
    elif shape in [
        (1024, 20480, 6144),
    ]:
        return [_config(128, 128, 32, 4, 4)]
    elif shape in [
        (2048, 10240, 25408),
    ]:
        return [_config(128, 128, 32, 16, 4)]
    elif shape in [
        (1024, 6144, 4096),
    ]:
        return [_config(128, 256, 64, 8, 8)]
    return [_config(128, 128, 32, 1, 4)]


#: Element multiple Triton's divisibility specialisation looks for.
_STRIDE_ALIGN = 16

#: Symbolic knobs of optimizations
_ALIGN_ROWS = 1
_PEEL_K_TAIL = 1


def _env_flag(name):
    value = os.environ.get(name, "1")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return value == "1"


def _align_rows(t):
    """Repack a row-major operand whose row stride defeats vectorised loads.

    Triton applies its divisibility-by-16 check to the raw stride *integer*, so
    a row stride like 1894 marks the pointer as arbitrarily aligned and every
    operand load drops from `buffer_load_dwordx4` (16 B) to `buffer_load_ushort`
    (2 B) -- 8x narrower, for the whole K loop. Verified by holding K fixed and
    changing only the stride: 1894 gives 16 ushort + 2 dwordx4 per iteration,
    1904 gives 4 dwordx4.
    """
    if t.stride(1) != 1 or t.stride(0) % _STRIDE_ALIGN == 0:
        return t
    rows, cols = t.shape
    padded = torch.empty((rows, triton.cdiv(cols, _STRIDE_ALIGN) * _STRIDE_ALIGN), device=t.device, dtype=t.dtype)
    padded[:, :cols] = t
    return padded[:, :cols]


def _prune_configs(configs, named_args, **kwargs):
    """Drop tiles that cannot run on this shape before anything is compiled."""
    elem_bytes = named_args["a_ptr"].element_size()
    kept = []
    for config in configs:
        kw = config.kwargs
        if lds_bytes(kw["BLOCK_M"], kw["BLOCK_N"], kw["BLOCK_K"], 1, elem_bytes) <= CDNA3_LDS_BYTES:
            kept.append(config)
    if not kept:
        raise RuntimeError(f"No config fits within the {CDNA3_LDS_BYTES} B gfx942 LDS budget")
    return kept


@functools.lru_cache(maxsize=None)
def _tuned(space, shape=None):
    """Autotuned kernel per search space; `shape` keys only the heuristic one."""
    if space == "heuristic":
        configs = heuristic_config(*shape) or SMOKE_CONFIGS()
    else:
        configs = {"full": CONFIGS, "smoke": SMOKE_CONFIGS}[space]()
    # EVEN_K has to be a heuristic rather than a launch argument because it
    # depends on BLOCK_K, which autotune has not chosen yet at the call site.
    kernel = triton.heuristics({"EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0})(matmul_kernel_gfx942)
    return triton.autotune(
        configs=configs,
        key=["M", "N", "K", "PEEL_K_TAIL", "ALIGN_ROWS"],
        prune_configs_by={"early_config_prune": _prune_configs},
    )(kernel)


def mm(a, b, *, space="heuristic"):
    """Matrix multiply ``a @ b`` on MI300X.

    `space` selects the search space -- "full" for perf, "heuristic" (one
    config) for a first call that stays interactive, "smoke" for path coverage.
    Not exposed on `tlx.ops.mm`.

    Either operand may be column-major: the kernel indexes through explicit
    strides, so a transposed view costs nothing and needs no copy.
    """
    assert a.shape[1] == b.shape[0], f"K mismatch: A={tuple(a.shape)}, B={tuple(b.shape)}"
    assert a.dtype == b.dtype, "A and B must have the same dtype"
    if _ALIGN_ROWS:
        a, b = _align_rows(a), _align_rows(b)
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )  # noqa: E731
    kernel = _tuned(space, (M, N, K) if space == "heuristic" else None)
    kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        PEEL_K_TAIL=_PEEL_K_TAIL,
        # Host-side only, but carried as constexpr metadata so autotuning keys
        # aligned and unaligned experiments independently.
        ALIGN_ROWS=_ALIGN_ROWS,
        # 16x16x16 MFMA on gfx942 fp16. kpack stays at its default of 1 -- see
        # the module docstring for the miscompile that follows from raising it.
        matrix_instr_nonkdim=16,
    )
    return c


matmul = mm
