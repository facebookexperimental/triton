"""MI350X (gfx950/CDNA4) ``tlx.ops.addmm`` implementation."""

import torch

from ..mm import gfx950 as _mm

_OUTPUT_PRECISION_BIAS_SHAPES = {
    (torch.float16, 1_024, 20_480, 6_144),
    (torch.float16, 3_072, 10_240, 128),
    (torch.float16, 3_072, 4_096, 2_048),
    (torch.float16, 3_072, 8_192, 2_048),
    (torch.float16, 3_072, 65_536, 768),
    (torch.float16, 4_096, 4_096, 2_048),
    (torch.float16, 61_440, 10_000, 1_024),
}


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
    if a.dtype == torch.float16 and shape == (61_440, 5_120, 7_744):
        return _mm._launch_dispatch(
            a,
            b,
            out,
            ("inter_wave", {"TILE": (256, 256), "SPLIT_K": 1}),
            bias=bias,
        )
    if a.dtype == torch.float16 and shape in (
        (3_072, 5_888, 1_536),
        (3_072, 5_888, 2_304),
    ):
        config = _mm._fixed_register_plan(64, 128, 64, 16, 8, 4, 2)
        if shape == (3_072, 5_888, 2_304):
            config = dict(config, OUTPUT_PRECISION_BIAS=True)
        return _mm._launch_register_plan(
            a,
            b,
            bias=bias,
            config=config,
            out=out,
        )
    if a.dtype == torch.float16 and shape == (3_072, 10_240, 128) and bias.stride(0) == 0:
        return _mm._launch_register_plan(
            a,
            b,
            bias=bias,
            config=_mm._fixed_register_plan(128, 128, 128, 8, 8, 4, 2, matrix_instr_nonkdim=32),
            out=out,
        )
    dispatch = _mm._dispatch_for(a, b)
    if dispatch is None:
        return _launch_uncovered_shape(a, b, bias, out)
    if shape == (4_096, 2_048, 4_096):
        dispatch = "wave_grid", _mm._wg_regular_wave_grid_plan(
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
    if a.dtype == torch.float16 and shape == (3_072, 65_536, 768) and bias.stride(0) == 0:
        dispatch = dispatch[0], dict(dispatch[1] or {}, GROUP_M_OVERRIDE=8)
    if (a.dtype, *shape) in _OUTPUT_PRECISION_BIAS_SHAPES and dispatch[0] == "inter_wave" and bias.stride(0) == 0:
        dispatch = dispatch[0], dict(dispatch[1] or {}, OUTPUT_PRECISION_BIAS=True)
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
