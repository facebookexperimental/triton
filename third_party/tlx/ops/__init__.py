"""TLX op library.

    from triton.tlx.ops import mm as tlx_mm
    c = tlx_mm(a, b)

    from triton.tlx.ops import addmm as tlx_addmm
    c = tlx_addmm(bias, a, b)

This module is the API contract; everything under it is private -- reaching into
`triton.tlx.ops.kernels.*` is not supported. Exactly one implementation ships
per (op, arch), so there is no `variant=` argument, and architecture never
appears in caller code.

Dispatch derives the architecture from the input tensor's device. If that
architecture cannot be determined, or the op has no implementation for it, the
call raises `UnsupportedOp`. The `space=` keyword selects an
implementation-defined autotune search space. Passing a space that the selected
implementation does not provide raises `InvalidInput` rather than silently
choosing another space. `mm` and `addmm` also accept `out=` for implementations
that support a preallocated output.

`space=` defaults to "heuristic" -- a single config chosen analytically -- for
any op that offers one, so that a first call stays interactive. The exception
is `mm` on sm90, whose full space is only 8 configs; it defaults to "full".
Measured on
B200, `mm` at `space="full"` takes 221-285s on a cold Triton cache (348 configs
compiled and benchmarked for a 1024x1024x1024 product) and also accumulates
tens of GB of autotune workspaces; at "heuristic" the same call is under a
second. On implementations that provide it, pass `space="full"` explicitly to
buy back the tuned configs, which are worth up to ~4x on small shapes. The
gfx950 implementation currently provides only `"heuristic"`.

Ops with no heuristic yet -- flash_attn, flash_attn_mxfp8, hstu_attn,
kimi_delta_attention -- still default to "full". Their remaining space is "smoke", which selects for
lowering-path coverage rather than speed, so defaulting to it would quietly
ship a bad config. Each needs its own `heuristic_config` before it can follow
`mm`. `mm_mxfp8` has a heuristic but defaults to "full" because its tuned
configs are measurably faster; pass `space="heuristic"` for a fast first call.

An op with no implementation for the current GPU raises `UnsupportedOp` -- it
never falls back to torch. A forward-only implementation raises
`UnsupportedBackward` before launch when autograd is enabled and any supported
tensor input requires gradients; inference under `torch.no_grad()` is allowed.
"""

from __future__ import annotations

from ._catalog import InvalidInput, UnsupportedBackward, UnsupportedOp, check_backward, check_inputs, impl_for

__all__ = [
    "mm", "mm_mxfp8", "grouped_gemm", "grouped_gemm_mxfp8", "addmm", "flash_attn", "flash_attn_mxfp8",
    "flash_attn_varlen", "hstu_attn_dev", "kimi_delta_attention", "kda_paged_prefill", "kda_recurrent_decode",
    "UnsupportedOp", "UnsupportedBackward", "InvalidInput"
]


def mm(a, b, *, out=None, space=None):
    """`a @ b`, for `(M, K) @ (K, N)` fp16/bf16. Either operand may be column-major.

    This op is currently forward-only.

    `space=None` uses the implementation's default: "full" on sm90, otherwise
    a single analytically chosen config so the first call stays interactive.
    Pass `space="full"` to implementations that expose a full autotune space;
    unsupported spaces raise `InvalidInput`. See the module docstring.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise InvalidInput("tlx.ops.mm expects two rank-2 tensors; "
                           f"got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}")
    if a.shape[1] != b.shape[0]:
        raise InvalidInput("tlx.ops.mm reduction dimensions must match; "
                           f"got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}")
    if a.dtype != b.dtype or a.device != b.device:
        raise InvalidInput("tlx.ops.mm operands must have the same dtype and device; "
                           f"got a=({a.dtype}, {a.device}), b=({b.dtype}, {b.device})")
    fn, spec = impl_for("mm", device=a.device)
    if spec.accepts is None:
        check_inputs(spec, dtype=a.dtype)
    else:
        # Mirror the kernel's operand prep: a non-contiguous operand is fed to
        # its descriptor transposed, so that is the stride TMA must find aligned.
        a_src = a if a.is_contiguous() else a.T
        b_src = b if b.is_contiguous() else b.T
        check_inputs(spec, dtype=a.dtype, M=a.shape[0], N=b.shape[1], K=a.shape[1],
                     row_strides=(a_src.stride(0), b_src.stride(0), b.shape[1]), elem_bytes=a.element_size())
    check_backward(spec, a, b)
    if space is None:
        space = spec.default_space
    if out is None:
        return fn(a, b, space=space)
    return fn(a, b, out=out, space=space)


def _check_mm_mxfp8_scales(a_scale, b_scale, *, M, N, K, sf_layout):
    if sf_layout == "natural":
        expected_a, expected_b = (M, K // 32), (N, K // 32)
        if a_scale.shape != expected_a or b_scale.shape != expected_b:
            raise InvalidInput("tlx.ops.mm_mxfp8 natural scales must have exact shapes "
                               f"a_scale={expected_a}, b_scale={expected_b}; got "
                               f"a_scale={tuple(a_scale.shape)}, b_scale={tuple(b_scale.shape)}")
        return
    if sf_layout != "cublas_blocked":
        raise InvalidInput("tlx.ops.mm_mxfp8 sf_layout must be 'natural' or 'cublas_blocked'; "
                           f"got {sf_layout!r}")
    # M, N and K are 128-aligned, so the 128x4 atoms need no padding.
    if a_scale.numel() != M * K // 32 or b_scale.numel() != N * K // 32:
        raise InvalidInput("tlx.ops.mm_mxfp8 cublas_blocked scales must contain exactly "
                           f"{M * K // 32} a-scale and {N * K // 32} b-scale bytes; "
                           f"got {a_scale.numel()} and {b_scale.numel()}")


def mm_mxfp8(a, a_scale, b, b_scale, *, out=None, sf_layout="natural", space="full"):
    """``a @ b.T`` over pre-quantized MXFP8 inputs, returning BF16 ``[M, N]``.

    ``a`` is contiguous E4M3 ``[M, K]`` and ``b`` is contiguous E4M3 ``[N, K]``
    (K-major, as for a linear weight); M, N and K must be multiples of 128.
    Each run of 32 values along K has one E8M0 scale. ``sf_layout="natural"``
    takes ``[M, K // 32]`` and ``[N, K // 32]`` scales; ``"cublas_blocked"``
    takes the swizzled byte layout returned by torchao's
    ``MXTensor.to_mx(..., is_swizzled_scales=True)``. A supplied ``out`` is
    returned by identity. Forward-only.

    ``space`` defaults to "full" (autotune; the first call per shape compiles
    and benchmarks the pruned space). "heuristic" launches one shape-picked
    config without autotuning.
    """
    import torch

    tensors = (a, a_scale, b, b_scale)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise InvalidInput("tlx.ops.mm_mxfp8 expects tensor inputs")
    if a.ndim != 2 or b.ndim != 2:
        raise InvalidInput("tlx.ops.mm_mxfp8 expects rank-2 a and b; "
                           f"got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}")
    M, K = a.shape
    N, b_k = b.shape
    if b_k != K:
        raise InvalidInput("tlx.ops.mm_mxfp8 reduction dimensions must match; "
                           f"got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}")
    if min(M, N, K) <= 0 or M % 128 or N % 128 or K % 128:
        raise InvalidInput("tlx.ops.mm_mxfp8 requires positive M, N and K divisible by 128; "
                           f"got M={M}, N={N}, K={K}")
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise InvalidInput("tlx.ops.mm_mxfp8 expects E4M3 a and b; "
                           f"got a.dtype={a.dtype}, b.dtype={b.dtype}")
    if a_scale.dtype != torch.float8_e8m0fnu or b_scale.dtype != torch.float8_e8m0fnu:
        raise InvalidInput("tlx.ops.mm_mxfp8 expects E8M0 a_scale and b_scale; "
                           f"got a_scale.dtype={a_scale.dtype}, b_scale.dtype={b_scale.dtype}")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise InvalidInput("tlx.ops.mm_mxfp8 expects all inputs to be contiguous")
    if a.device.type != "cuda" or any(tensor.device != a.device for tensor in tensors[1:]):
        raise InvalidInput("tlx.ops.mm_mxfp8 expects all inputs on the same CUDA device; "
                           f"got {[tensor.device for tensor in tensors]}")
    _check_mm_mxfp8_scales(a_scale, b_scale, M=M, N=N, K=K, sf_layout=sf_layout)
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise InvalidInput("tlx.ops.mm_mxfp8 expects out to be a tensor or None")
        if out.shape != (M, N) or out.dtype != torch.bfloat16 or out.device != a.device or not out.is_contiguous():
            raise InvalidInput("tlx.ops.mm_mxfp8 out must be contiguous BF16 [M, N] on a's device; "
                               f"got shape={tuple(out.shape)}, dtype={out.dtype}, device={out.device}")
        if any(torch._C._overlaps(out, tensor) for tensor in tensors):
            raise InvalidInput("tlx.ops.mm_mxfp8 out must not overlap any input")
    if space not in ("heuristic", "full"):
        raise InvalidInput(f"tlx.ops.mm_mxfp8 does not provide space={space!r}")

    fn, spec = impl_for("mm_mxfp8", device=a.device)
    tma_tensors = tensors if out is None else (*tensors, out)
    check_inputs(spec, dtype=a.dtype, base_ptrs=tuple(tensor.data_ptr() for tensor in tma_tensors))
    check_backward(spec, *tensors, out)
    return fn(a, a_scale, b, b_scale, out=out, sf_layout=sf_layout, space=space)


def grouped_gemm(group_a, group_b):
    """Run a ragged group of FP16 ``A[i] @ B[i]`` products.

    ``A[i]`` must be rank-2 row-major. ``B[i]`` must be a rank-2 column-major
    view with K contiguous. The operation is currently forward-only;
    architecture-specific alignment restrictions may also apply.
    """
    import torch

    if not isinstance(group_a, (list, tuple)) or not isinstance(group_b, (list, tuple)):
        raise InvalidInput("tlx.ops.grouped_gemm expects group_a and group_b to be lists or tuples")
    group_a = tuple(group_a)
    group_b = tuple(group_b)
    if len(group_a) != len(group_b):
        raise InvalidInput("tlx.ops.grouped_gemm operand groups must have the same length; "
                           f"got len(group_a)={len(group_a)}, len(group_b)={len(group_b)}")
    if not group_a:
        raise InvalidInput("tlx.ops.grouped_gemm requires at least one matrix pair")

    for index, (a, b) in enumerate(zip(group_a, group_b)):
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise InvalidInput("tlx.ops.grouped_gemm expects tensor elements; "
                               f"pair {index} has a={type(a).__name__}, b={type(b).__name__}")
        if a.ndim != 2 or b.ndim != 2:
            raise InvalidInput("tlx.ops.grouped_gemm expects rank-2 tensors; "
                               f"pair {index} has a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}")
        if any(dim <= 0 for dim in (*a.shape, *b.shape)):
            raise InvalidInput("tlx.ops.grouped_gemm dimensions must be positive; "
                               f"pair {index} has a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}")
        if a.shape[1] != b.shape[0]:
            raise InvalidInput("tlx.ops.grouped_gemm reduction dimensions must match; "
                               f"pair {index} has a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)}")
        if a.stride() != (a.shape[1], 1):
            raise InvalidInput("tlx.ops.grouped_gemm requires row-major A tensors; "
                               f"pair {index} has a.stride={a.stride()}")
        if b.stride() != (1, b.shape[0]):
            raise InvalidInput("tlx.ops.grouped_gemm requires column-major B tensors; "
                               f"pair {index} has b.stride={b.stride()}")

    dtype = group_a[0].dtype
    device = group_a[0].device
    for index, tensor in enumerate((*group_a, *group_b)):
        if tensor.dtype != dtype or tensor.device != device:
            raise InvalidInput("tlx.ops.grouped_gemm operands must share a dtype and device; "
                               f"tensor {index} is ({tensor.dtype}, {tensor.device}), expected ({dtype}, {device})")

    fn, spec = impl_for("grouped_gemm", device=device)
    # TMA implementations describe A, the zero-copy B.T view, and C as
    # row-major tensors. Direct-load implementations ignore these facts.
    row_strides = tuple(stride for a, b in zip(group_a, group_b) for stride in (a.stride(0), b.stride(1), b.shape[1]))
    base_ptrs = tuple(tensor.data_ptr() for pair in zip(group_a, group_b) for tensor in pair)
    check_inputs(
        spec,
        dtype=dtype,
        row_strides=row_strides,
        base_ptrs=base_ptrs,
        elem_bytes=group_a[0].element_size(),
    )
    check_backward(spec, *group_a, *group_b)
    return fn(group_a, group_b)


def _grouped_gemm_mxfp8_dims(x, w, split_sizes):
    GM, K = x.shape
    G = split_sizes.shape[0]
    if w.ndim == 3:
        w_groups, N, w_k = w.shape
        if w_groups != G:
            raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 expects w.shape[0] == G; "
                               f"got w.shape={tuple(w.shape)}, G={G}")
    else:
        grouped_n, w_k = w.shape
        if G == 0 or grouped_n % G != 0:
            raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 expects packed w.shape[0] divisible by G; "
                               f"got w.shape={tuple(w.shape)}, G={G}")
        N = grouped_n // G
    return GM, G, N, K, w_k


def _check_grouped_gemm_mxfp8_scales(x_scale, w_scale, w, *, GM, G, N, K, sf_layout):
    scale_k = K // 32
    if sf_layout == "natural":
        expected_x = (GM, scale_k)
        expected_w = (G, N, scale_k) if w.ndim == 3 else (G * N, scale_k)
        if x_scale.shape != expected_x or w_scale.shape != expected_w:
            raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 natural scales must have exact shapes "
                               f"x_scale={expected_x}, w_scale={expected_w}; got "
                               f"x_scale={tuple(x_scale.shape)}, w_scale={tuple(w_scale.shape)}")
        return

    if sf_layout != "cublas_blocked":
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 sf_layout must be 'natural' or "
                           f"'cublas_blocked'; got {sf_layout!r}")
    if x_scale.ndim != 2 or w_scale.ndim != 2:
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 cublas_blocked scales must be rank-2 byte tensors; "
                           f"got x_scale.ndim={x_scale.ndim}, w_scale.ndim={w_scale.ndim}")
    padded_scale_k = ((scale_k + 3) // 4) * 4
    expected_x_bytes = ((GM + 127) // 128) * 128 * padded_scale_k
    expected_w_bytes = G * ((N + 127) // 128) * 128 * padded_scale_k
    x_bytes = x_scale.numel() * x_scale.element_size()
    w_bytes = w_scale.numel() * w_scale.element_size()
    if x_bytes != expected_x_bytes or w_bytes != expected_w_bytes:
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 cublas_blocked scales must contain exactly "
                           f"{expected_x_bytes} x-scale bytes and {expected_w_bytes} w-scale bytes; "
                           f"got {x_bytes} and {w_bytes}")


def _check_grouped_gemm_mxfp8_out(out, tensors, *, GM, N, device):
    import torch

    if out is None:
        return
    if not isinstance(out, torch.Tensor):
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 expects out to be a tensor or None")
    if out.shape != (GM, N) or out.dtype != torch.bfloat16 or out.device != device or not out.is_contiguous():
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 out must be contiguous BF16 [GM, N] on x's device; "
                           f"got shape={tuple(out.shape)}, dtype={out.dtype}, device={out.device}")
    if any(torch._C._overlaps(out, tensor) for tensor in tensors):
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 out must not overlap any input")


def grouped_gemm_mxfp8(x, x_scale, w, w_scale, split_sizes, *, out=None, num_sms=None, sf_layout="natural"):
    """Run a forward-only SM100 or gfx950 grouped GEMM over pre-quantized MXFP8 inputs.

    ``x`` is contiguous ``[GM, K]`` and ``w`` is contiguous ``[G, N, K]``
    or packed ``[G * N, K]`` E4M3 data. Natural E8M0 scales match those
    logical data shapes with a final ``K // 32`` dimension. CuBLAS-blocked
    scales are rank-2 opaque byte tensors with exact 128x4-atom storage.

    ``split_sizes`` stays on device: the kernel validates that values are
    nonnegative, every prefix is 128-aligned, and the final sum is ``GM``.
    The BF16 result has shape ``[GM, N]``; a supplied ``out`` is returned by
    identity. ``num_sms`` defaults to all SMs (CUs on AMD) on ``x.device``.
    """
    import torch

    tensors = (x, x_scale, w, w_scale, split_sizes)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 expects tensor inputs")
    if x.ndim != 2 or w.ndim not in (2, 3) or split_sizes.ndim != 1:
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 expects x/w/split_sizes ranks 2/(2 or 3)/1; "
                           f"got {x.ndim}/{w.ndim}/{split_sizes.ndim}")

    GM, G, N, K, w_k = _grouped_gemm_mxfp8_dims(x, w, split_sizes)
    if min(GM, G, N, K) <= 0:
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 requires positive GM, G, N, and K; "
                           f"got GM={GM}, G={G}, N={N}, K={K}")
    if w_k != K:
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 reduction dimensions must match; "
                           f"got x.shape={tuple(x.shape)}, w.shape={tuple(w.shape)}")
    if GM % 128 != 0 or K % 128 != 0:
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 requires GM and K divisible by 128; "
                           f"got GM={GM}, K={K}")

    if x.dtype != torch.float8_e4m3fn or w.dtype != torch.float8_e4m3fn:
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 expects E4M3 x and w; "
                           f"got x.dtype={x.dtype}, w.dtype={w.dtype}")
    if x_scale.dtype != torch.float8_e8m0fnu or w_scale.dtype != torch.float8_e8m0fnu:
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 expects E8M0 x_scale and w_scale; "
                           f"got x_scale.dtype={x_scale.dtype}, w_scale.dtype={w_scale.dtype}")
    if split_sizes.dtype != torch.int32:
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 expects split_sizes.dtype == torch.int32; "
                           f"got {split_sizes.dtype}")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 expects all inputs to be contiguous")
    if x.device.type != "cuda" or any(tensor.device != x.device for tensor in tensors[1:]):
        raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 expects all inputs on the same CUDA device; "
                           f"got {[tensor.device for tensor in tensors]}")

    _check_grouped_gemm_mxfp8_scales(x_scale, w_scale, w, GM=GM, G=G, N=N, K=K, sf_layout=sf_layout)
    if num_sms is not None:
        device_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
        if type(num_sms) is not int or not 1 <= num_sms <= device_sms:
            raise InvalidInput("tlx.ops.grouped_gemm_mxfp8 num_sms must be an int in "
                               f"[1, {device_sms}] for {x.device}; got {num_sms!r}")
    _check_grouped_gemm_mxfp8_out(out, tensors, GM=GM, N=N, device=x.device)

    fn, spec = impl_for("grouped_gemm_mxfp8", device=x.device)
    tma_tensors = (x, x_scale, w, w_scale) if out is None else (x, x_scale, w, w_scale, out)
    check_inputs(
        spec,
        dtype=x.dtype,
        row_bytes=(
            x.stride(0) * x.element_size(),
            w.stride(-2) * w.element_size(),
            N * torch.bfloat16.itemsize,
        ),
        base_ptrs=tuple(tensor.data_ptr() for tensor in tma_tensors),
    )
    check_backward(spec, x, x_scale, w, w_scale, split_sizes, out)
    return fn(
        x,
        x_scale,
        w,
        w_scale,
        split_sizes,
        out=out,
        num_sms=num_sms,
        sf_layout=sf_layout,
    )


def addmm(input, a, b, *, out=None, space="heuristic"):
    """Fused ``input + a @ b`` for two-dimensional fp16/bf16 matrices.

    ``input`` may be ``(N,)`` or two-dimensional and broadcastable to the
    ``(M, N)`` result. Matrix and input scale factors are both one. This op is
    currently forward-only.
    """
    fn, spec = impl_for("addmm", device=a.device)
    check_inputs(spec, dtype=a.dtype)
    check_backward(spec, input, a, b)
    return fn(input, a, b, out=out, space=space)


def flash_attn(q, k, v, causal=False, sm_scale=None, *, space="full"):
    """Fused attention over `(Z, H, N_CTX, HEAD_DIM)` fp16/bf16. Differentiable.

    `sm_scale` defaults to `HEAD_DIM ** -0.5`.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise InvalidInput("tlx.ops.flash_attn expects rank-4 Q/K/V tensors")
    if space not in ("full", "smoke"):
        raise InvalidInput(f"tlx.ops.flash_attn does not provide space={space!r}")
    fn, spec = impl_for("flash_attn", device=q.device)
    check_inputs(spec, dtype=q.dtype, HEAD_DIM=q.shape[-1])
    check_backward(spec, q, k, v)
    # Triton's launcher uses the process' current device rather than deriving it
    # from pointer arguments. Keep compilation and launch on the input device;
    # the context manager restores the caller's current device afterwards.
    import torch
    with torch.cuda.device(q.device):
        return fn(q, k, v, causal, sm_scale, space=space)


def flash_attn_mxfp8(q, k, v, causal=False, sm_scale=None, *, space="full"):
    """MXFP8 attention over contiguous BF16 ``(Z, H, N_CTX, HEAD_DIM)`` tensors.

    Q, K, V and the softmax probabilities are quantized internally to E4M3
    data with E8M0 scales: per 32x32 block for Q and K, per 32 keys for V and
    the probabilities. gfx950 quantizes as Blackwell does, but its non-causal
    kernel computes the probabilities with an approximate exp2 (see
    kernels/flash_attn_mxfp8/gfx950.py), and the two agree to within the
    quantization error rather than bitwise. Head dim 128 and sequence lengths
    divisible by 256. Blackwell returns BF16 gradients; gfx950 is forward only.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise InvalidInput("tlx.ops.flash_attn_mxfp8 expects rank-4 Q/K/V tensors")
    if q.shape != k.shape or q.shape != v.shape:
        raise InvalidInput("tlx.ops.flash_attn_mxfp8 expects Q/K/V to have identical shapes; "
                           f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.device != k.device or q.device != v.device:
        raise InvalidInput("tlx.ops.flash_attn_mxfp8 expects Q/K/V to have the same dtype and device")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise InvalidInput("tlx.ops.flash_attn_mxfp8 expects contiguous Q/K/V tensors")
    if space not in ("full", "smoke"):
        raise InvalidInput(f"tlx.ops.flash_attn_mxfp8 does not provide space={space!r}")
    fn, spec = impl_for("flash_attn_mxfp8", device=q.device)
    check_inputs(spec, dtype=q.dtype, HEAD_DIM=q.shape[-1], N_CTX=q.shape[-2])
    check_backward(spec, q, k, v)
    return fn(q, k, v, causal, sm_scale, space=space)


def flash_attn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, sm_scale=None, causal=False):
    """Forward-only attention over packed ``(tokens, heads, HEAD_DIM)`` fp16/bf16 sequences.

    Sequence ``b`` owns query rows ``cu_seqlens_q[b]:cu_seqlens_q[b + 1]`` and
    key/value rows ``cu_seqlens_k[b]:cu_seqlens_k[b + 1]``; both offset tensors
    are int32 ``(batch + 1,)``. ``max_seqlen_q`` must bound every query length
    (it sizes the grid); ``max_seqlen_k`` is accepted for flash-attn API
    parity. K/V may have fewer heads than Q (GQA) if they divide its head count.
    Causal masking is bottom-right aligned: query ``i`` sees keys
    ``j <= i + k_len - q_len``, and a query row that sees no key returns zeros.
    `sm_scale` defaults to `HEAD_DIM ** -0.5`. Returns the output only (no
    log-sum-exp). Output rows not covered by ``cu_seqlens_q`` are left
    uninitialized.
    """
    import torch
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise InvalidInput("tlx.ops.flash_attn_varlen expects rank-3 (tokens, heads, head_dim) Q/K/V tensors")
    if k.shape != v.shape or q.shape[2] != k.shape[2] or q.shape[1] % k.shape[1] != 0:
        raise InvalidInput("tlx.ops.flash_attn_varlen expects K/V of identical shape, Q's head dim, and a head count "
                           f"dividing Q's; got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise InvalidInput("tlx.ops.flash_attn_varlen expects Q/K/V to have the same dtype")
    if q.stride(2) != 1 or k.stride(2) != 1 or v.stride(2) != 1:
        raise InvalidInput("tlx.ops.flash_attn_varlen expects unit-stride head dims")
    if (cu_seqlens_q.ndim != 1 or cu_seqlens_q.shape != cu_seqlens_k.shape or cu_seqlens_q.shape[0] < 1
            or cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32):
        raise InvalidInput("tlx.ops.flash_attn_varlen expects int32 (batch + 1,) cu_seqlens_q and cu_seqlens_k")
    if any(t.device != q.device for t in (k, v, cu_seqlens_q, cu_seqlens_k)):
        raise InvalidInput("tlx.ops.flash_attn_varlen expects all tensors on one device")
    fn, spec = impl_for("flash_attn_varlen", device=q.device)
    check_inputs(spec, dtype=q.dtype, HEAD_DIM=q.shape[2])
    check_backward(spec, q, k, v)
    with torch.cuda.device(q.device):
        return fn(q, k, v, cu_seqlens_q, cu_seqlens_k, int(max_seqlen_q), sm_scale, bool(causal))


def hstu_attn_dev(q, k, v, seq_offsets, max_seq_len, attn_scale, alpha=None, causal=True, num_targets=None,
                  max_attn_len=0, contextual_seq_len=0, *, space="full"):
    """HSTU ragged attention over `(total_tokens, H, HEAD_DIM)` fp16/bf16. Differentiable.

    Scores are SiLU-scaled rather than softmaxed, which is why this is its own
    op. `seq_offsets` is `(B + 1,)` prefix offsets; `alpha` defaults to
    `1 / HEAD_DIM`.

    Causal-only: `causal=False` raises `InvalidInput`. The argument is kept so
    the intent is stated at the call site rather than assumed.
    """
    fn, spec = impl_for("hstu_attn_dev", device=q.device)
    check_inputs(spec, dtype=q.dtype, HEAD_DIM=q.shape[-1], causal=causal)
    check_backward(spec, q, k, v)
    return fn(q, k, v, seq_offsets, max_seq_len, alpha if alpha is not None else 1.0 / q.shape[-1], causal=causal,
              attn_scale=attn_scale, num_targets=num_targets, max_attn_len=max_attn_len,
              contextual_seq_len=contextual_seq_len, space=space)


def kimi_delta_attention(q, k, v, g, beta, *, scale=1.0, cu_seqlens=None, cu_seqlens_cpu=None, space="full"):
    """Kimi Delta Attention over packed `[1, T, H, 128]` fp16/bf16 inputs.

    Returns the TritonBench-compatible `(output, None)` pair.
    """
    fn, spec = impl_for("kimi_delta_attention", device=q.device)
    check_inputs(spec, dtype=q.dtype, HEAD_DIM=q.shape[-1])
    check_backward(spec, q, k, v, g, beta)
    return fn(q, k, v, g, beta, scale=scale, cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, space=space)


def kda_paged_prefill(q, k, v, g, beta, *, scale=1.0, initial_state, cu_seqlens):
    """Prepared-input chunked KDA prefill for packed `[1, T, H, 128]` tensors.

    `g` contains per-channel log decays and `beta` is sigmoid-applied.
    State is FP32 and V-major: `[N, H, 128, 128]`. This op is currently
    forward-only.
    """
    fn, spec = impl_for("kda_paged_prefill", device=q.device)
    check_inputs(spec, dtype=q.dtype, KEY_DIM=q.shape[-1], VALUE_DIM=v.shape[-1])
    check_backward(spec, q, k, v, g, beta, initial_state)
    return fn(q, k, v, g, beta, scale=scale, initial_state=initial_state, cu_seqlens=cu_seqlens)


def kda_recurrent_decode(q, k, v, g, beta, *, scale=1.0, state_pool, read_indices, write_indices, cu_seqlens):
    """Forward-only indexed KDA recurrence over an FP32 V-major state pool."""
    fn, spec = impl_for("kda_recurrent_decode", device=q.device)
    check_inputs(spec, dtype=q.dtype, KEY_DIM=q.shape[-1], VALUE_DIM=v.shape[-1])
    check_backward(spec, q, k, v, g, beta, state_pool)
    return fn(q, k, v, g, beta, scale=scale, state_pool=state_pool, read_indices=read_indices,
              write_indices=write_indices, cu_seqlens=cu_seqlens)
