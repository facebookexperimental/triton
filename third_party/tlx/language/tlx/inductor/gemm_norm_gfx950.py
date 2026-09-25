"""Shared gfx950 addmm + normalization kernels and registration helpers."""

from __future__ import annotations

import functools
import torch
import triton
import triton.language as tl
from torch._inductor import config
from torch._inductor.pattern_matcher import Match
from torch.library import wrap_triton
from triton.tlx.ops.kernels.mm import gfx950 as gfx950_mm

from ..hw.target import current_target


_BLOCK_K = 64
_GEMM_BLOCK_N = 128
_NORM_BLOCK_N = 4096
_SPLIT_REDUCE_BLOCK_M = 32
_SPLIT_REDUCE_BLOCK_N = 256
_EPILOGUE_STATS = "epilogue_stats"
_SPLIT_STATS = "split_stats"
_SPLIT_ROW_NORM = "split_row_norm"


def _register_plans() -> tuple[tuple[object, ...], ...]:
    plans = []
    for triton_config in gfx950_mm._REGISTER_CONFIGS:
        kwargs = triton_config.kwargs
        plans.append((
            "register",
            kwargs["BLOCK_M"],
            kwargs["BLOCK_N"],
            kwargs["BLOCK_K"],
            kwargs["GROUP_M"],
            kwargs["NUM_XCDS"],
            1,
            triton_config.num_warps,
            triton_config.num_stages,
            kwargs["matrix_instr_nonkdim"],
            kwargs["waves_per_eu"],
            kwargs["kpack"],
            False,
        ))
    return tuple(dict.fromkeys(plans))


def _plan_split_k(plan: tuple[object, ...]) -> int:
    return int(plan[6])


_REGISTER_PLANS = _register_plans()


def _make_lds_plans(
    tile_plans: tuple[tuple[int, int, int], ...],
) -> tuple[tuple[object, ...], ...]:
    """Expand compact (block_m, block_n, split_k) choices into GEMM plans."""
    return tuple(
        (
            "lds",
            block_m,
            block_n,
            64,
            group_m,
            num_xcds,
            split_k,
            4 if block_m == 128 else 8,
            1,
            16,
            0,
            1,
            True,
        )
        for block_m, block_n, split_k in tile_plans
        for group_m in (1, 4, 8)
        for num_xcds in (1, 8)
    )


@triton.jit
def tlx_gfx950_addmm_norm_stats(
    workspace_ptr,
    gemm_bias_ptr,
    raw_ptr,
    row_sum_ptr,
    row_sum_sq_ptr,
    M,
    N,
    stride_gemm_bias,
    stride_raw_m,
    stride_raw_n,
    SPLIT_K: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    offsets = rows[:, None] * N + cols[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for split_k in range(SPLIT_K):
        acc += tl.load(
            workspace_ptr + split_k * M * N + offsets,
            mask=mask,
            other=0.0,
        )
    gemm_bias = tl.load(
        gemm_bias_ptr + cols * stride_gemm_bias,
        mask=cols < N,
        other=0.0,
    )
    value = (acc + gemm_bias[None, :]).to(tl.bfloat16)
    raw_offsets = rows[:, None] * stride_raw_m + cols[None, :] * stride_raw_n
    tl.store(raw_ptr + raw_offsets, value, mask=mask)

    stats_offsets = rows * N_BLOCKS + pid_n
    value_fp32 = value.to(tl.float32)
    if not IS_RMS_NORM:
        tl.store(
            row_sum_ptr + stats_offsets,
            tl.sum(value_fp32, axis=1),
            mask=rows < M,
        )
    tl.store(
        row_sum_sq_ptr + stats_offsets,
        tl.sum(value_fp32 * value_fp32, axis=1),
        mask=rows < M,
    )


@triton.jit
def tlx_gfx950_apply_norm(
    raw_ptr,
    row_sum_ptr,
    row_sum_sq_ptr,
    scale_ptr,
    norm_bias_ptr,
    output_ptr,
    M,
    stride_raw_m,
    stride_raw_n,
    stride_scale,
    stride_norm_bias,
    stride_output_m,
    stride_output_n,
    EPS: tl.constexpr,
    N: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    USE_PARTIAL_STATS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_STATS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < N
    raw_offsets = row * stride_raw_m + cols * stride_raw_n
    value = tl.load(raw_ptr + raw_offsets, mask=col_mask, other=0.0).to(
        tl.float32
    )
    if USE_PARTIAL_STATS:
        stats_cols = tl.arange(0, BLOCK_STATS)
        stats_mask = stats_cols < N_BLOCKS
        stats_offsets = row * N_BLOCKS + stats_cols
        row_sum_sq = tl.sum(
            tl.load(row_sum_sq_ptr + stats_offsets, mask=stats_mask, other=0.0),
            axis=0,
        )
    else:
        row_sum_sq = tl.sum(value * value, axis=0)
    if IS_RMS_NORM:
        mean = 0.0
        inverse_std = tl.rsqrt(row_sum_sq / N + EPS)
    else:
        if USE_PARTIAL_STATS:
            row_sum = tl.sum(
                tl.load(
                    row_sum_ptr + stats_offsets,
                    mask=stats_mask,
                    other=0.0,
                ),
                axis=0,
            )
        else:
            row_sum = tl.sum(value, axis=0)
        mean = row_sum / N
        variance = tl.maximum(row_sum_sq / N - mean * mean, 0.0)
        inverse_std = tl.rsqrt(variance + EPS)

    scale = tl.load(
        scale_ptr + cols * stride_scale,
        mask=col_mask,
    ).to(tl.float32)
    normalized = (value - mean) * inverse_std * scale
    if not IS_RMS_NORM:
        normalized += tl.load(
            norm_bias_ptr + cols * stride_norm_bias,
            mask=col_mask,
        ).to(tl.float32)
    output_offsets = row * stride_output_m + cols * stride_output_n
    tl.store(output_ptr + output_offsets, normalized, mask=col_mask)


@triton.jit
def tlx_gfx950_addmm_norm_row_reduce(
    workspace_ptr,
    gemm_bias_ptr,
    scale_ptr,
    norm_bias_ptr,
    output_ptr,
    M,
    stride_workspace_m,
    stride_workspace_n,
    stride_gemm_bias,
    stride_scale,
    stride_norm_bias,
    stride_output_m,
    stride_output_n,
    EPS: tl.constexpr,
    N: tl.constexpr,
    SPLIT_K: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < N
    acc = tl.zeros((BLOCK_N, ), dtype=tl.float32)
    for split_k in range(SPLIT_K):
        workspace_row = split_k * M + row
        acc += tl.load(
            workspace_ptr + workspace_row * stride_workspace_m + cols * stride_workspace_n,
            mask=col_mask,
            other=0.0,
        )
    acc += tl.load(
        gemm_bias_ptr + cols * stride_gemm_bias,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)

    value = acc.to(output_ptr.dtype.element_ty).to(tl.float32)
    row_sum_sq = tl.sum(value * value, axis=0)
    if IS_RMS_NORM:
        mean = 0.0
        inverse_std = tl.rsqrt(row_sum_sq / N + EPS)
    else:
        mean = tl.sum(value, axis=0) / N
        variance = tl.maximum(row_sum_sq / N - mean * mean, 0.0)
        inverse_std = tl.rsqrt(variance + EPS)

    scale = tl.load(scale_ptr + cols * stride_scale, mask=col_mask).to(
        tl.float32
    )
    normalized = (value - mean) * inverse_std * scale
    if not IS_RMS_NORM:
        normalized += tl.load(
            norm_bias_ptr + cols * stride_norm_bias,
            mask=col_mask,
        ).to(tl.float32)
    output_offsets = row * stride_output_m + cols * stride_output_n
    tl.store(output_ptr + output_offsets, normalized, mask=col_mask)


def _launch_gfx950_addmm_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    gemm_bias: torch.Tensor,
    scale: torch.Tensor,
    norm_bias: torch.Tensor,
    eps: float,
    *,
    is_rms_norm: bool,
    gemm_plan: tuple[object, ...],
    norm_impl: str,
    norm_num_warps: int,
    apply_block_n: int = _NORM_BLOCK_N,
    stats_block_m: int = _SPLIT_REDUCE_BLOCK_M,
    stats_block_n: int = _SPLIT_REDUCE_BLOCK_N,
    stats_num_warps: int = 4,
) -> torch.Tensor:
    m, k = x.shape
    n = weight.shape[1]
    (
        kind,
        block_m,
        block_n,
        block_k,
        group_size_m,
        num_xcds,
        split_k,
        num_warps,
        num_stages,
        matrix_instr_nonkdim,
        waves_per_eu,
        kpack,
        disable_agpr,
    ) = gemm_plan
    grid_mn = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    raw = torch.empty((m, n), device=x.device, dtype=x.dtype)
    use_partial_stats = norm_impl != _SPLIT_ROW_NORM and (
        split_k > 1 or norm_impl == _EPILOGUE_STATS
    )
    partial_block_n = stats_block_n if split_k > 1 else block_n
    n_blocks = triton.cdiv(n, partial_block_n)
    row_sum = (
        raw
        if not use_partial_stats or is_rms_norm
        else torch.empty(
            (m, n_blocks),
            device=x.device,
            dtype=torch.float32,
        )
    )
    row_sum_sq = (
        raw
        if not use_partial_stats
        else torch.empty(
            (m, n_blocks),
            device=x.device,
            dtype=torch.float32,
        )
    )
    workspace = (
        raw
        if split_k == 1
        else torch.empty(
            (split_k * m, n),
            device=x.device,
            dtype=torch.float32,
        )
    )

    if kind == "register":
        launch_options = {}
        if disable_agpr:
            launch_options["llvm_fn_attrs"] = (
                ("amdgpu-agpr-alloc", "0,0"), )
        wrap_triton(gfx950_mm._register_kernel_impl)[(grid_mn, )](
            x,
            weight,
            gemm_bias,
            raw,
            row_sum,
            row_sum_sq,
            m,
            n,
            k,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            weight.stride(1),
            0,
            gemm_bias.stride(0),
            raw.stride(0),
            raw.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=group_size_m,
            NUM_XCDS=num_xcds,
            ADD_BIAS=True,
            WRITE_STATS=use_partial_stats,
            IS_RMS_NORM=is_rms_norm,
            num_warps=num_warps,
            num_stages=num_stages,
            matrix_instr_nonkdim=matrix_instr_nonkdim,
            waves_per_eu=waves_per_eu,
            kpack=kpack,
            **launch_options,
        )
    else:
        uneven_split_k = k % split_k != 0
        wrap_triton(gfx950_mm.a16w16_8wave)[
            (grid_mn * split_k, )
        ](
            x,
            weight,
            gemm_bias,
            raw,
            workspace,
            row_sum,
            row_sum_sq,
            m,
            n,
            k,
            k // split_k,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            weight.stride(1),
            0,
            gemm_bias.stride(0),
            workspace.stride(0),
            workspace.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_SIZE_M=group_size_m,
            NUM_XCDS=num_xcds,
            GRID_MN=grid_mn,
            SPLIT_K=split_k,
            ADD_BIAS=split_k == 1,
            HAS_REGISTER_TAIL=(
                uneven_split_k or (k // split_k) % (2 * block_k) != 0
            ),
            USE_I64_A_OFFSETS=False,
            USE_I64_B_OFFSETS=False,
            USE_I64_C_OFFSETS=False,
            UNEVEN_SPLIT_K=uneven_split_k,
            HAS_M_TAIL=m % block_m != 0,
            HAS_N_TAIL=n % block_n != 0,
            PIN_OFFSET_LAYOUT=False,
            DEFER_EPILOGUE=split_k > 1,
            WRITE_STATS=use_partial_stats,
            IS_RMS_NORM=is_rms_norm,
            num_warps=num_warps,
            num_stages=num_stages,
            matrix_instr_nonkdim=matrix_instr_nonkdim,
            llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
            enable_sched_group_barrier_scheduler=True,
        )

    if split_k > 1 and norm_impl != _SPLIT_ROW_NORM:
        wrap_triton(tlx_gfx950_addmm_norm_stats)[
            (
                triton.cdiv(m, stats_block_m),
                n_blocks,
            )
        ](
            workspace,
            gemm_bias,
            raw,
            row_sum,
            row_sum_sq,
            m,
            n,
            gemm_bias.stride(0),
            raw.stride(0),
            raw.stride(1),
            SPLIT_K=split_k,
            N_BLOCKS=n_blocks,
            BLOCK_M=stats_block_m,
            BLOCK_N=stats_block_n,
            IS_RMS_NORM=is_rms_norm,
            num_warps=stats_num_warps,
        )

    if norm_impl == _SPLIT_ROW_NORM:
        output = torch.empty_like(raw)
        wrap_triton(tlx_gfx950_addmm_norm_row_reduce)[(m, )](
            workspace,
            gemm_bias,
            scale,
            norm_bias,
            output,
            m,
            workspace.stride(0),
            workspace.stride(1),
            gemm_bias.stride(0),
            scale.stride(0),
            norm_bias.stride(0),
            output.stride(0),
            output.stride(1),
            EPS=eps,
            N=n,
            SPLIT_K=split_k,
            IS_RMS_NORM=is_rms_norm,
            BLOCK_N=triton.next_power_of_2(n),
            num_warps=norm_num_warps,
        )
        return output

    output = torch.empty_like(raw)
    apply_grid_n = triton.cdiv(n, apply_block_n) if use_partial_stats else 1
    wrap_triton(tlx_gfx950_apply_norm)[(m, apply_grid_n)](
        raw,
        row_sum,
        row_sum_sq,
        scale,
        norm_bias,
        output,
        m,
        raw.stride(0),
        raw.stride(1),
        scale.stride(0),
        norm_bias.stride(0),
        output.stride(0),
        output.stride(1),
        EPS=eps,
        N=n,
        N_BLOCKS=n_blocks,
        IS_RMS_NORM=is_rms_norm,
        USE_PARTIAL_STATS=use_partial_stats,
        BLOCK_N=apply_block_n,
        BLOCK_STATS=triton.next_power_of_2(n_blocks),
        num_warps=norm_num_warps,
    )
    return output


def _eligible(match: Match, expected_shape: tuple[int, int, int]) -> bool:
    if config.triton.tlx_mode not in ("allow", "force"):
        return False
    if not current_target().is_gfx950:
        return False
    addmm = next(
        (
            node
            for node in match.nodes
            if node.op == "call_function" and node.target == torch.ops.aten.addmm.default
        ),
        None,
    )
    if addmm is None or len(addmm.users) != 1:
        return False
    tensor_names = ["x", "weight", "gemm_bias", "scale"]
    if "norm_bias" in match.kwargs:
        tensor_names.append("norm_bias")
    tensors = [
        match.kwargs[name].meta.get("val")
        for name in tensor_names
    ]
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        return False
    x, weight, gemm_bias, scale, *optional_norm_bias = tensors
    if x.ndim != 2 or weight.ndim != 2:
        return False
    vector_inputs = [gemm_bias, scale, *optional_norm_bias]
    if any(value.ndim != 1 for value in vector_inputs):
        return False
    try:
        m = int(x.shape[0])
        k = int(x.shape[1])
        weight_k = int(weight.shape[0])
        n = int(weight.shape[1])
        vector_sizes = [int(value.shape[0]) for value in vector_inputs]
        x_strides = tuple(int(stride) for stride in x.stride())
        weight_strides = tuple(int(stride) for stride in weight.stride())
        vector_strides = [int(value.stride(0)) for value in vector_inputs]
    except (TypeError, ValueError):
        return False
    return bool(
        (m, k, n) == expected_shape
        and x.dtype == torch.bfloat16
        and weight.dtype == x.dtype
        and gemm_bias.dtype == x.dtype
        and scale.dtype == x.dtype
        and all(value.dtype == x.dtype for value in optional_norm_bias)
        and all(value.device == x.device for value in tensors)
        and k == weight_k
        and all(size == n for size in vector_sizes)
        and x_strides == (k, 1)
        and weight_strides == (1, k)
        and all(stride == 1 for stride in vector_strides)
        and n % _GEMM_BLOCK_N == 0
        and k % _BLOCK_K == 0
    )


def _candidate_configs(
    CustomOpConfig,
    fused_impl,
    *,
    lds_plans: tuple[tuple[object, ...], ...],
    focused_plans: tuple[tuple[object, ...], ...],
    is_rms_norm: bool,
):
    configs = []
    # Exercise every register-resident plan maintained by tlx.ops. The GEMM
    # epilogue emits partial row statistics while its accumulators are live.
    for plan in _REGISTER_PLANS:
        configs.append(
            CustomOpConfig(
                fused_impl,
                gemm_plan=plan,
                norm_impl=_EPILOGUE_STATS,
                norm_num_warps=8,
            )
        )

    # Direct-to-LDS candidates cover both tile geometry and split-K. A split-K
    # plan fuses bias and partial statistics into its workspace reducer; a
    # non-split plan emits its partial statistics directly from the GEMM.
    for plan in lds_plans:
        split_k = _plan_split_k(plan)
        configs.append(
            CustomOpConfig(
                fused_impl,
                gemm_plan=plan,
                norm_impl=_SPLIT_STATS if split_k > 1 else _EPILOGUE_STATS,
                norm_num_warps=8,
            )
        )
        if split_k > 1:
            for norm_num_warps in (2, 4, 8, 16):
                configs.append(
                    CustomOpConfig(
                        fused_impl,
                        gemm_plan=plan,
                        norm_impl=_SPLIT_ROW_NORM,
                        norm_num_warps=norm_num_warps,
                    )
                )

    # Search normalization occupancy for the strongest measured GEMM plans.
    for plan in focused_plans:
        norm_impl = (
            _SPLIT_STATS
            if _plan_split_k(plan) > 1
            else _EPILOGUE_STATS
        )
        for apply_block_n in (256, 512, 1024, 2048, 4096):
            for norm_num_warps in (2, 4, 8):
                configs.append(
                    CustomOpConfig(
                        fused_impl,
                        gemm_plan=plan,
                        norm_impl=norm_impl,
                        norm_num_warps=norm_num_warps,
                        apply_block_n=apply_block_n,
                    )
                )
        if is_rms_norm and _plan_split_k(plan) > 1:
            for stats_block_m in (16, 32, 64):
                for stats_block_n in (128, 256):
                    for stats_num_warps in (4, 8):
                        configs.append(
                            CustomOpConfig(
                                fused_impl,
                                gemm_plan=plan,
                                norm_impl=_SPLIT_STATS,
                                norm_num_warps=8,
                                stats_block_m=stats_block_m,
                                stats_block_n=stats_block_n,
                                stats_num_warps=stats_num_warps,
                            )
                        )
    return configs


def _register_autotuned_region(
    custom_op,
    fused_impl,
    aten_impl,
    name: str,
    *,
    lds_plans: tuple[tuple[object, ...], ...],
    focused_plans: tuple[tuple[object, ...], ...],
    is_rms_norm: bool,
) -> None:
    from torch._inductor.kernel.custom_op import (
        CustomOpConfig,
        register_custom_op_autotuning,
    )
    from torch._inductor.lowering import user_lowerings

    fused_configs = _candidate_configs(
        CustomOpConfig,
        fused_impl,
        lds_plans=lds_plans,
        focused_plans=focused_plans,
        is_rms_norm=is_rms_norm,
    )
    register_custom_op_autotuning(
        custom_op,
        configs=[*fused_configs, CustomOpConfig(aten_impl)],
        name=f"{name}_allow",
        include_fallback=False,
    )
    op_overload = custom_op._opoverload
    allow_lowering = user_lowerings[op_overload]

    register_custom_op_autotuning(
        custom_op,
        configs=fused_configs,
        name=name,
        include_fallback=False,
    )
    force_lowering = user_lowerings[op_overload]

    @functools.wraps(allow_lowering)
    def lowering(*args, **kwargs):
        if config.triton.tlx_mode == "force":
            return force_lowering(*args, **kwargs)
        return allow_lowering(*args, **kwargs)

    user_lowerings[op_overload] = lowering
