"""MI350X (gfx950/CDNA4) ``tlx.ops.addmm`` implementation."""

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from ..mm import gfx950 as _mm

_SHALLOW_K_4WAVE_ACC_LAYOUT = tlx.layout(
    shape=((16, 4, 2, 2), (4, 4, 8)),
    stride=((128, 4, 16, 2048), (1, 32, 4096)),
)
_SHALLOW_K_4WAVE_STORE_LAYOUT = tlx.layout(
    shape=((64, 4), (8, 16)),
    stride=((8, 512), (1, 2048)),
)
_SHALLOW_K192_FRAGMENT_A_LAYOUT = tlx.padded_shared_layout_encoding.with_bases(
    [(512, 16)],
    _mm._iw_swz_offset_bases([256, 64], 1),
    [256, 64],
)
_SHALLOW_K192_CACHED_B_HALF_LAYOUT = tlx.padded_shared_layout_encoding.with_bases(
    [(512, 16)],
    _mm._iw_swz_offset_bases([64, 128], 0),
    [64, 128],
)
_SHALLOW_K256_MT256_PLAN = _mm._wg_regular_wave_grid_plan(
    8,
    8,
    2,
    2,
    64,
    local_stages=2,
    pgr2_operands=True,
    row_wise_epilogue=True,
    wide_epilogue=True,
    num_xcds=1,
    workgroup_mapping=1,
    llvm_sched_strategy="iterative-ilp",
    dot_mode="logical",
)
_SHALLOW_K256_MT256_PLAN.update(
    store_cache=".cs",
    fused_group_fragments=8,
    prefetch_bias=True,
)
_SHALLOW_K512_MT256_DIRECT_PLAN = dict(
    _SHALLOW_K256_MT256_PLAN,
    direct_to_lds=True,
    direct_chunk=32,
    pack_direct_chunks=True,
    row_major_b_lds=True,
    workgroup_mapping=4,
    loop_unroll_factor=2,
    dot_mode="persistent",
    output_precision_bias=True,
    bias_cache=".cg",
    prefetch_bias=False,
)
_SHALLOW_K768_BF16_VECTOR_PLAN = dict(
    _SHALLOW_K512_MT256_DIRECT_PLAN,
    pack_direct_chunks=False,
    loop_unroll_factor=3,
    dot_mode="persistent",
    enable_sched_group_barrier_scheduler=True,
    sched_group_barrier_mfma_per_dwordx4=6,
    output_precision_bias=False,
    bias_cache="",
    prefetch_bias=True,
)
_SHALLOW_K768_FP16_VECTOR_PLAN = dict(
    _SHALLOW_K768_BF16_VECTOR_PLAN,
    num_xcds=8,
    workgroup_mapping=8,
    refill_read_window=1,
    pack_direct_chunks=True,
)
_SHALLOW_K768_BF16_MATRIX_PLAN = dict(
    _SHALLOW_K768_BF16_VECTOR_PLAN,
    loop_unroll_factor=3,
    workgroup_mapping=4,
    output_precision_bias=True,
    bias_cache=".cg",
    prefetch_matrix_bias=True,
    matrix_bias_prefetch_rows=2,
    prefetch_bias=False,
    llvm_fn_attrs=(("amdgpu-sched-strategy", "iterative-maxocc"), ),
)
_SHALLOW_K768_FP16_MATRIX_PLAN = dict(
    _SHALLOW_K512_MT256_DIRECT_PLAN,
    loop_unroll_factor=3,
    prefetch_matrix_bias=True,
    matrix_bias_prefetch_rows=3,
    llvm_fn_attrs=(("amdgpu-sched-strategy", "iterative-maxocc"), ),
)
_SHALLOW_K128_MT256_PLAN = _mm._wg_regular_wave_grid_plan(
    8,
    4,
    2,
    2,
    32,
    local_stages=1,
    pgr2_operands=True,
    row_wise_epilogue=True,
    wide_epilogue=True,
    num_xcds=1,
    workgroup_mapping=1,
    llvm_sched_strategy="iterative-ilp",
    dot_mode="logical",
)
_SHALLOW_K128_MT256_PLAN.update(
    store_cache=".cs",
    fused_group_fragments=4,
    prefetch_bias=True,
    llvm_fn_attrs="",
)
_SHALLOW_K128_MATRIX_PLAN = dict(
    _SHALLOW_K128_MT256_PLAN,
    mi_wave_tile_m=6,
    block_k=64,
    disable_unclustered_high_rp_reschedule=True,
    dot_mode="persistent",
    output_precision_bias=True,
    bias_cache=".cg",
    prefetch_matrix_bias=True,
    matrix_bias_prefetch_rows=4,
    prefetch_bias=False,
)
_SHALLOW_K80_FRAGMENTED_PLAN = _mm._wg_register_fragmented_m_plan(
    (16, 64),
    256,
    128,
    num_warps=4,
    num_stages=2,
)
_SHALLOW_K359_MT32X64_PLAN = _mm._wg_regular_wave_grid_plan(
    1,
    2,
    2,
    2,
    64,
    local_stages=1,
    pgr2_operands=True,
    row_wise_epilogue=True,
    wide_epilogue=True,
    num_xcds=1,
    workgroup_mapping=1,
    llvm_sched_strategy="iterative-ilp",
    dot_mode="logical",
    waves_per_eu=2,
)
_SHALLOW_K359_MT32X64_PLAN.update(
    local_stages=2,
    compact_k64_prefetch_before_publish=True,
    exact_lds=True,
    store_cache=".cs",
    fused_group_fragments=2,
    prefetch_bias=True,
    llvm_fn_attrs=(("amdgpu-sched-strategy", "iterative-maxocc"), ),
    regclass_priority_trumps_globalness=False,
    sink_insts_to_avoid_spills=True,
    disable_unclustered_high_rp_reschedule=True,
)
_SHALLOW_K797_MT192X128_PLAN = _mm._wg_regular_wave_grid_plan(
    6,
    4,
    2,
    2,
    64,
    local_stages=2,
    pgr2_operands=True,
    direct_to_lds=True,
    direct_chunk=32,
    pack_direct_chunks=True,
    loop_unroll_factor=3,
    read_cover_sixteenths=11,
    row_wise_epilogue=True,
    wide_epilogue=True,
    num_xcds=1,
    workgroup_mapping=4,
    reverse_local_assignment=False,
    sink_insts_to_avoid_spills=True,
    disable_unclustered_high_rp_reschedule=True,
    enable_sched_group_barrier_scheduler=True,
    sched_group_barrier_mfma_per_dwordx4=4,
    exact_lds=False,
    dot_mode="logical",
    llvm_sched_strategy="iterative-ilp",
    waves_per_eu=1,
)
_SHALLOW_K797_MT192X128_PLAN.update(
    fused_group_fragments=4,
    store_cache=".cs",
    prefetch_bias=True,
)
_SHALLOW_K607_PADDED_DISPATCH_OPTIONS = {
    "OUTPUT_PRECISION_BIAS": True,
    "STORE_CACHE": ".cs",
    "BIAS_CACHE": ".cg",
    "GROUP_M_OVERRIDE": 6,
    "VECTOR_BIAS_BROADCAST": True,
}

_OUTPUT_PRECISION_BIAS_SHAPES = {
    (torch.float16, 1_024, 20_480, 6_144),
    (torch.float16, 3_072, 4_096, 2_048),
    (torch.float16, 3_072, 4_096, 7_424),
    (torch.float16, 3_072, 8_192, 2_048),
    (torch.float16, 4_096, 4_096, 2_048),
    (torch.float16, 61_440, 2_560, 4_096),
    (torch.float16, 61_440, 3_840, 4_096),
    (torch.float16, 61_440, 5_120, 2_048),
    (torch.float16, 61_440, 10_000, 1_024),
}

_REGISTER_OUTPUT_PRECISION_MATRIX_BIAS_SHAPES = {
    (torch.float16, 3_072, 40_440, 256),
    (torch.float16, 3_072, 44_536, 128),
}


@triton.jit
def _pack_shallow_k607_rows(
    source,
    destination,
    program,
    ROW_COUNT: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    rows = program * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    cols0 = tl.arange(0, 512)
    row_mask = rows[:, None] < ROW_COUNT
    values0 = tl.load(
        source + rows[:, None] * 607 + cols0[None, :],
        mask=row_mask,
        other=0.0,
        eviction_policy="evict_first",
    )
    tl.store(
        destination + rows[:, None] * 608 + cols0[None, :],
        values0,
        mask=row_mask,
    )
    cols1 = 512 + tl.arange(0, 128)
    values1 = tl.load(
        source + rows[:, None] * 607 + cols1[None, :],
        mask=row_mask & (cols1[None, :] < 607),
        other=0.0,
        eviction_policy="evict_first",
    )
    tl.store(
        destination + rows[:, None] * 608 + cols1[None, :],
        values1,
        mask=row_mask & (cols1[None, :] < 608),
    )


@triton.jit
def _pack_shallow_k607_kernel(
    a_src,
    b_src,
    a_dst,
    b_dst,
    A_ROWS: tl.constexpr,
    B_ROWS: tl.constexpr,
    A_PROGRAMS: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    program = tl.program_id(0)
    if program < A_PROGRAMS:
        _pack_shallow_k607_rows(a_src, a_dst, program, A_ROWS, ROWS_PER_PROGRAM)
    else:
        _pack_shallow_k607_rows(
            b_src,
            b_dst,
            program - A_PROGRAMS,
            B_ROWS,
            ROWS_PER_PROGRAM,
        )


@triton.jit
def _shallow_k_4wave_vector_bias_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    A_CACHE: tl.constexpr,
    N_TILES_PER_CTA: tl.constexpr,
    PREFETCH_NEXT_B: tl.constexpr,
):
    block_m: tl.constexpr = 256
    block_n: tl.constexpr = 256
    block_k: tl.constexpr = 64
    half_n: tl.constexpr = 128
    num_xcds: tl.constexpr = 8
    grid_m: tl.constexpr = M // block_m
    n_groups: tl.constexpr = (N // block_n) // N_TILES_PER_CTA
    grid_mn: tl.constexpr = grid_m * n_groups

    program = tl.program_id(0)
    xcd = program % num_xcds
    linear = xcd * (grid_mn // num_xcds) + program // num_xcds
    pid_m = linear // n_groups
    pid_n_base = (linear % n_groups) * N_TILES_PER_CTA

    a_cache = tl.tuple([
        tlx.local_alloc(
            (block_m, block_k),
            tlx.dtype_of(a_ptr),
            1,
            layout=_SHALLOW_K192_FRAGMENT_A_LAYOUT,
        ) for _ in range(3)
    ])
    b_stage = tlx.local_alloc(
        (block_k, half_n),
        tlx.dtype_of(b_ptr),
        2,
        layout=_SHALLOW_K192_CACHED_B_HALF_LAYOUT,
    )
    rows = pid_m * block_m + tl.arange(0, block_m)
    reduction = tl.arange(0, block_k)
    a_offsets = rows[:, None] * stride_am + reduction[None, :] * stride_ak
    for kb in tl.static_range(3):
        tlx.buffer_load_to_local(
            tlx.local_view(a_cache[kb], 0),
            a_ptr,
            a_offsets + kb * block_k * stride_ak,
            cache_modifier=A_CACHE,
        )
    tlx.async_load_commit_group()
    tlx.async_load_wait_group(0)
    tl.debug_barrier()

    mma: tl.constexpr = tlx.amd_mfma_layout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[2, 2],
    )
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, mma, k_width=8)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, mma, k_width=8)

    for local_pid_n in tl.static_range(N_TILES_PER_CTA):
        pid_n = pid_n_base + local_pid_n
        for half in tl.static_range(2):
            cols = pid_n * block_n + half * half_n + tl.arange(0, half_n)
            b_offsets = reduction[:, None] * stride_bk + cols[None, :] * stride_bn
            acc = tlx.zeros((block_m, half_n), tl.float32, layout=mma)
            if not PREFETCH_NEXT_B or local_pid_n * 2 + half == 0:
                for stage in tl.static_range(2):
                    tlx.buffer_load_to_local(
                        tlx.local_view(b_stage, stage),
                        b_ptr,
                        b_offsets + stage * block_k * stride_bk,
                    )
                    tlx.async_load_commit_group()

            tlx.async_load_wait_group(1)
            a = tlx.local_load(
                tlx.local_view(a_cache[0], 0),
                layout=dot_a,
                relaxed=True,
            )
            b_value = tlx.local_load(
                tlx.local_view(b_stage, 0),
                layout=dot_b,
                relaxed=True,
            )
            tl.debug_barrier()
            acc = tl.dot(a, b_value, acc)
            tlx.buffer_load_to_local(
                tlx.local_view(b_stage, 0),
                b_ptr,
                b_offsets + 2 * block_k * stride_bk,
            )
            tlx.async_load_commit_group()

            tlx.async_load_wait_group(1)
            a = tlx.local_load(
                tlx.local_view(a_cache[1], 0),
                layout=dot_a,
                relaxed=True,
            )
            b_value = tlx.local_load(
                tlx.local_view(b_stage, 1),
                layout=dot_b,
                relaxed=True,
            )
            tl.debug_barrier()
            acc = tl.dot(a, b_value, acc)

            tlx.async_load_wait_group(0)
            a = tlx.local_load(
                tlx.local_view(a_cache[2], 0),
                layout=dot_a,
                relaxed=True,
            )
            b_value = tlx.local_load(
                tlx.local_view(b_stage, 0),
                layout=dot_b,
                relaxed=True,
            )
            acc = tl.dot(a, b_value, acc)
            # All waves must finish the K2 read before stage 0 can hold the
            # next N half, including the next loop iteration for BF16.
            tl.debug_barrier()
            if PREFETCH_NEXT_B and local_pid_n * 2 + half + 1 < N_TILES_PER_CTA * 2:
                next_pid_n = pid_n_base + (local_pid_n * 2 + half + 1) // 2
                next_cols = (next_pid_n * block_n + (local_pid_n * 2 + half + 1) % 2 * half_n + tl.arange(0, half_n))
                next_b_offsets = reduction[:, None] * stride_bk + next_cols[None, :] * stride_bn
                for stage in tl.static_range(2):
                    tlx.buffer_load_to_local(
                        tlx.local_view(b_stage, stage),
                        b_ptr,
                        next_b_offsets + stage * block_k * stride_bk,
                    )
                    tlx.async_load_commit_group()
            acc = tlx.amd_mfma_commit(acc)

            offsets = rows[:, None] * stride_cm + cols[None, :] * stride_cn
            offsets = tlx.require_layout(offsets, _SHALLOW_K_4WAVE_STORE_LAYOUT)
            accumulator = tlx.require_layout(acc, _SHALLOW_K_4WAVE_ACC_LAYOUT)
            bias = tl.load(bias_ptr + cols)[None, :]
            value = tlx.require_layout(
                accumulator.to(c_ptr.dtype.element_ty),
                _SHALLOW_K_4WAVE_STORE_LAYOUT,
            ) + bias
            tl.store(c_ptr + offsets, value, cache_modifier=".cs")


def _launch_shallow_k_4wave_vector_bias(a, b, bias, out):
    m, _ = a.shape
    n = b.shape[1]
    n_tiles_per_cta = 2 if a.dtype == torch.bfloat16 else 4
    grid = (m // 256 * (n // 256 // n_tiles_per_cta), )
    _shallow_k_4wave_vector_bias_kernel[grid](
        a,
        b,
        bias,
        out,
        m,
        n,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        A_CACHE=".cg",
        N_TILES_PER_CTA=n_tiles_per_cta,
        PREFETCH_NEXT_B=a.dtype == torch.float16,
        num_warps=4,
        num_stages=1,
        matrix_instr_nonkdim=16,
    )
    return out


def _launch_shallow_k192_cached_a_bias(a, b, bias, out):
    return _launch_shallow_k_4wave_vector_bias(a, b, bias, out)


def _launch_shallow_k256_mt256_bias(a, b, bias, out):
    plan = _SHALLOW_K256_MT256_PLAN
    if a.shape[0] == 64_000:
        plan = dict(
            plan,
            num_xcds=8,
            pgr2_late_read_count=1,
        )
    return _mm._wg_matmul(
        a,
        b,
        bias=bias,
        out=out,
        _candidate_plan=plan,
    )


def _launch_shallow_k512_mt256_matrix_bias(a, b, bias, out):
    plan = dict(
        _SHALLOW_K512_MT256_DIRECT_PLAN,
        prefetch_matrix_bias=True,
        matrix_bias_prefetch_rows=3,
    )
    if b.shape[1] in (19_584, 19_968):
        plan = dict(
            plan,
            mi_wave_tile_m=6,
            llvm_fn_attrs=(("amdgpu-sched-strategy", "iterative-maxocc"), ),
            tiles_per_program=1,
            linear_tiles=False,
            num_xcds=1,
            xcd_group_size=1,
            refill_read_window=1 if b.shape[1] == 19_968 else 0,
            enable_sched_group_barrier_scheduler=b.shape[1] == 19_968,
            sched_group_barrier_mfma_per_dwordx4=6 if b.shape[1] == 19_968 else 4,
        )
    elif b.shape[1] == 22_240:
        plan = dict(
            plan,
            matrix_bias_prefetch_rows=4,
        )
    return _mm._wg_matmul(
        a,
        b,
        bias=bias,
        out=out,
        _candidate_plan=plan,
    )


def _launch_shallow_k768_bf16_vector_bias(a, b, bias, out):
    return _mm._wg_matmul(
        a,
        b,
        bias=bias,
        out=out,
        _candidate_plan=_SHALLOW_K768_BF16_VECTOR_PLAN,
    )


def _launch_shallow_k768_fp16_vector_bias(a, b, bias, out):
    return _mm._wg_matmul(
        a,
        b,
        bias=bias,
        out=out,
        _candidate_plan=_SHALLOW_K768_FP16_VECTOR_PLAN,
    )


def _launch_shallow_k768_bf16_matrix_bias(a, b, bias, out):
    return _mm._wg_matmul(
        a,
        b,
        bias=bias,
        out=out,
        _candidate_plan=_SHALLOW_K768_BF16_MATRIX_PLAN,
    )


def _launch_shallow_k768_fp16_matrix_bias(a, b, bias, out):
    return _mm._wg_matmul(
        a,
        b,
        bias=bias,
        out=out,
        _candidate_plan=_SHALLOW_K768_FP16_MATRIX_PLAN,
    )


def _launch_shallow_k128_mt256_bias(a, b, bias, out):
    plan = _SHALLOW_K128_MT256_PLAN
    if bias.stride(0) != 0:
        plan = _SHALLOW_K128_MATRIX_PLAN
    elif (a.shape[0], b.shape[1], a.shape[1]) == (3_072, 10_240, 128):
        plan = dict(
            plan,
            dot_mode="persistent",
            output_precision_bias=True,
        )
    return _mm._wg_matmul(
        a,
        b,
        bias=bias,
        out=out,
        _candidate_plan=plan,
    )


def _launch_shallow_k80_fragmented_bias(a, b, bias, out):
    _mm._wg_matmul(
        b.T,
        a.T,
        bias=bias.T,
        out=out.T,
        _candidate_plan=_SHALLOW_K80_FRAGMENTED_PLAN,
    )
    return out


def _launch_shallow_k359_mt32x64_bias(a, b, bias, out):
    return _mm._wg_matmul(
        a,
        b,
        bias=bias,
        out=out,
        _candidate_plan=_SHALLOW_K359_MT32X64_PLAN,
    )


def _launch_shallow_k797_mt192x128_bias(a, b, bias, out):
    return _mm._wg_matmul(
        a,
        b,
        bias=bias,
        out=out,
        _candidate_plan=_SHALLOW_K797_MT192X128_PLAN,
    )


def _shallow_k607_padded_dispatch(a, b):
    dispatch = _mm._dispatch_for(a, b)
    if dispatch is None:
        raise RuntimeError("No gfx950 dispatch for padded shallow-K operands")
    return dispatch[0], dict(dispatch[1] or {}, **_SHALLOW_K607_PADDED_DISPATCH_OPTIONS)


def _launch_shallow_k607_padded(a, b, bias, out):
    m, _ = a.shape
    n = b.shape[1]
    a_padded = torch.empty((m, 608), device=a.device, dtype=a.dtype)
    b_storage = torch.empty((n, 608), device=b.device, dtype=b.dtype)
    b_padded = b_storage.T
    rows_per_program = 16
    a_programs = triton.cdiv(m, rows_per_program)
    b_programs = triton.cdiv(n, rows_per_program)
    grid = (a_programs + b_programs, )
    _pack_shallow_k607_kernel[grid](
        a,
        b,
        a_padded,
        b_storage,
        A_ROWS=m,
        B_ROWS=n,
        A_PROGRAMS=a_programs,
        ROWS_PER_PROGRAM=rows_per_program,
        num_warps=4,
    )
    dispatch = _shallow_k607_padded_dispatch(a_padded, b_padded)
    return _mm._launch_dispatch(a_padded, b_padded, out, dispatch, bias=bias)


def _validate_operands(a, b, out):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected A[M, K] and B[K, N], got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"K mismatch: A={tuple(a.shape)}, B={tuple(b.shape)}")
    if a.device.type != "cuda" or b.device != a.device:
        raise ValueError("A and B must be on the same GPU")
    if a.dtype != b.dtype:
        raise ValueError("A and B must have the same dtype")
    if any(stride <= 0 for stride in (*a.stride(), *b.stride())):
        raise ValueError("gfx950 addmm requires positive matrix strides")
    m, k = a.shape
    n = b.shape[1]
    if out is not None:
        if out.shape != (m, n) or out.device != a.device or out.dtype != a.dtype or not out.is_contiguous():
            raise ValueError(f"out must be a contiguous {a.dtype} tensor with shape ({m}, {n}) on A's device")
    return m, n, k


def _expanded_input(input, m, n, a):
    if input.device != a.device or input.dtype != a.dtype:
        raise ValueError("input must match A's device and dtype")
    if input.ndim == 1:
        if input.shape[0] != n:
            raise ValueError(f"1-D addmm input must have shape ({n},), got {tuple(input.shape)}")
        return input.reshape(1, n).expand(m, n)
    if input.ndim == 2 and input.shape[0] in (1, m) and input.shape[1] in (1, n):
        return input.expand(m, n)
    raise ValueError(f"addmm input with shape {tuple(input.shape)} is not broadcastable to ({m}, {n})")


def _launch_uncovered_shape(a, b, bias, out):
    config = _mm._register_plan_for_shape(a.shape[0], b.shape[1], a.shape[1], a.dtype)
    if config is None:
        config = _mm._intermediate_register_config(a.shape[0], b.shape[1], a.shape[1])
    return _mm._launch_register_plan(a, b, bias=bias, config=config, out=out)


def _launch(a, b, bias, out):
    shape = (a.shape[0], b.shape[1], a.shape[1])
    if (a.dtype == torch.float16 and shape in ((64_000, 256, 256), (2_252_800, 256, 256)) and a.stride() == (256, 1)
            and b.stride() == (1, 256) and bias.stride() == (0, 1)):
        return _launch_shallow_k256_mt256_bias(a, b, bias, out)
    if (a.dtype == torch.float16 and shape[0] == 3_072 and shape[1] in (19_584, 19_968, 22_240) and shape[2] == 512
            and a.stride() == (512, 1) and b.stride() == (1, 512) and bias.stride() == (shape[1], 1)):
        return _launch_shallow_k512_mt256_matrix_bias(a, b, bias, out)
    if (a.dtype == torch.bfloat16 and shape == (2_048, 40_192, 768) and a.stride() == (768, 1)
            and b.stride() == (1, 768) and bias.stride() == (0, 1)):
        return _launch_shallow_k768_bf16_vector_bias(a, b, bias, out)
    if (a.dtype == torch.float16 and shape == (3_072, 65_536, 768) and a.stride() == (768, 1) and b.stride() == (1, 768)
            and bias.stride() == (0, 1)):
        return _launch_shallow_k768_fp16_vector_bias(a, b, bias, out)
    if (a.dtype == torch.bfloat16 and shape == (1_024, 47_232, 768) and a.stride() == (768, 1)
            and b.stride() == (1, 768) and bias.stride() == (47_232, 1)):
        return _launch_shallow_k768_bf16_matrix_bias(a, b, bias, out)
    if (a.dtype == torch.float16 and shape in ((3_072, 19_584, 768), (3_072, 39_936, 768)) and a.stride() == (768, 1)
            and b.stride() == (1, 768) and bias.stride() == (shape[1], 1)):
        return _launch_shallow_k768_fp16_matrix_bias(a, b, bias, out)
    if (a.dtype == torch.float16 and shape == (3_072, 10_240, 128) and a.stride() == (128, 1) and b.stride() == (1, 128)
            and bias.stride() == (0, 1)):
        return _launch_shallow_k128_mt256_bias(a, b, bias, out)
    if (a.dtype == torch.float16 and shape == (3_072, 44_032, 128) and a.stride() == (128, 1) and b.stride() == (1, 128)
            and bias.stride() == (44_032, 1)):
        return _launch_shallow_k128_mt256_bias(a, b, bias, out)
    if (a.dtype == torch.float16 and shape == (4_718_592, 80, 80) and a.stride() == (80, 1) and b.stride() == (1, 80)
            and bias.stride() == (0, 1)):
        return _launch_shallow_k80_fragmented_bias(a, b, bias, out)
    if (a.dtype == torch.float16 and shape == (256, 2_048, 359) and a.stride() == (359, 1) and b.stride() == (1, 359)
            and bias.stride() == (0, 1)):
        return _launch_shallow_k359_mt32x64_bias(a, b, bias, out)
    if (a.dtype == torch.float16 and shape == (3_072, 2_048, 797) and a.stride() == (797, 1) and b.stride() == (1, 797)
            and bias.stride() == (0, 1)):
        return _launch_shallow_k797_mt192x128_bias(a, b, bias, out)
    if (a.dtype == torch.float16 and shape == (819_200, 1_024, 192) and a.stride() == (192, 1)
            and b.stride() == (1, 192) and bias.stride() == (0, 1)):
        return _launch_shallow_k192_cached_a_bias(a, b, bias, out)
    if (a.dtype == torch.bfloat16 and shape == (819_200, 1_024, 192) and a.stride() == (192, 1)
            and b.stride() == (1, 192) and bias.stride() == (0, 1)):
        return _launch_shallow_k192_cached_a_bias(a, b, bias, out)
    if (a.dtype == torch.float16 and shape == (3_072, 38_848, 607) and a.stride() == (607, 1) and b.stride() == (1, 607)
            and bias.stride() == (0, 1)):
        return _launch_shallow_k607_padded(a, b, bias, out)
    if a.dtype == torch.float16 and shape == (61_440, 5_120, 7_744):
        return _mm._launch_dispatch(
            a,
            b,
            out,
            ("inter_wave", {"TILE": (256, 256), "SPLIT_K": 1, "STORE_CACHE": ".cs", "OUTPUT_PRECISION_BIAS": True}),
            bias=bias,
        )
    if a.dtype == torch.float16 and shape in (
        (3_072, 5_888, 1_536),
        (3_072, 5_888, 2_304),
    ):
        config = dict(
            _mm._fixed_register_plan(64, 128, 64, 16, 8, 4, 2),
            BIAS_CACHE=".cg",
            BIAS_EVICTION_POLICY="evict_first",
        )
        if shape == (3_072, 5_888, 2_304):
            config = dict(config, OUTPUT_PRECISION_BIAS=True)
        return _mm._launch_register_plan(
            a,
            b,
            bias=bias,
            config=config,
            out=out,
        )
    dispatch = _mm._dispatch_for(a, b)
    if dispatch is None:
        return _launch_uncovered_shape(a, b, bias, out)
    if ((a.dtype, *shape) in _REGISTER_OUTPUT_PRECISION_MATRIX_BIAS_SHAPES and dispatch[0] == "register"
            and bias.stride(0) != 0):
        options = dict(
            dispatch[1],
            OUTPUT_PRECISION_BIAS=True,
            BIAS_CACHE=".cg",
            BIAS_EVICTION_POLICY="evict_first",
        )
        if shape == (3_072, 40_440, 256):
            options["STORE_CACHE"] = ".cs"
        dispatch = dispatch[0], options
    if shape == (4_096, 2_048, 4_096):
        plan = _mm._wg_regular_wave_grid_plan(
            4,
            8,
            2,
            2,
            64,
            local_stages=2,
            pgr2_operands=True,
            direct_to_lds=True,
            row_major_b_lds=True,
            row_wise_epilogue=True,
            wide_epilogue=True,
            read_cover_sixteenths=7,
            loop_unroll_factor=2,
            num_xcds=1,
            workgroup_mapping=4,
            reverse_local_assignment=True,
            sink_insts_to_avoid_spills=True,
            dot_mode="logical",
            llvm_sched_strategy="iterative-ilp",
            waves_per_eu=1,
        )
        dispatch = "wave_grid", dict(plan, fused_group_fragments=4)
    if a.dtype == torch.float16 and shape == (3_072, 4_096, 797) and bias.stride(0) != 0:
        dispatch = dispatch[0], dict(
            dispatch[1],
            mi_wave_tile_m=6,
            wide_epilogue=True,
            fused_group_fragments=8,
            output_precision_bias=True,
            store_cache=".cs",
            bias_cache=".cg",
            prefetch_matrix_bias=True,
            matrix_bias_prefetch_rows=6,
            local_stages=2,
            direct_to_lds=False,
            exact_lds=True,
            pgr2_late_read_count=10,
            pack_direct_chunks=False,
            dot_mode="logical",
            llvm_fn_attrs=(("amdgpu-sched-strategy", "iterative-maxocc"), ),
            enable_sched_group_barrier_scheduler=False,
            sched_group_barrier_mfma_per_dwordx4=4,
        )
    if (a.dtype, *shape) in _OUTPUT_PRECISION_BIAS_SHAPES and dispatch[0] == "inter_wave" and bias.stride(0) == 0:
        dispatch = dispatch[0], dict(dispatch[1] or {}, OUTPUT_PRECISION_BIAS=True)
    if (dispatch[0] in ("wave_grid", "transposed_wave_grid") and bias.stride(0) == 0
            and dispatch[1].get("dot_mode", "persistent") != "transient"):
        dispatch = dispatch[0], dict(dispatch[1], prefetch_bias=True)
    return _mm._launch_dispatch(a, b, out, dispatch, bias=bias)


def addmm(input, a, b, *, out=None, space="heuristic"):
    """Compute ``input + a @ b`` with fused gfx950 GEMM epilogues.

    ``input`` may be ``(N,)`` or a two-dimensional tensor broadcastable to
    ``(M, N)``. Matrix and input scale factors are both one.
    """
    if space not in ("heuristic", "full", "smoke"):
        raise ValueError(f"Unknown gfx950 addmm search space: {space!r}")
    m, n, k = _validate_operands(a, b, out)
    bias = _expanded_input(input, m, n, a)
    result = out if out is not None else torch.empty((m, n), device=a.device, dtype=a.dtype)
    if m == 0 or n == 0:
        return result
    if k == 0:
        result.copy_(bias)
        return result
    return _launch(a, b, bias, result)
