import triton.language.core as tl

from . import types as tlx
from .utility import cuda_parse_arch


def require_nv_mma_shared_layout(x: tlx.buffered_tensor, swizzled: bool, _builder=None):
    assert isinstance(x.type.layout, tlx.DummySMemLayoutEncoding), "input must be a shared tensor"

    # Create a placeholder that will be resolved after inlining
    layout = tlx.DummySMemLayoutEncoding(list(x.shape), x.dtype)
    layout_handle = layout.to_ir(_builder)
    return _builder.create_require_layout(x.handle, layout_handle)


# async dot signature needs to be close to tl.dot as much as possible
@tl.builtin
def async_dot(
    A: tlx.buffered_tensor | tl.tensor,
    B: tlx.buffered_tensor,
    acc: tlx.buffered_tensor | tl.tensor | None = None,
    use_acc: (tl.constexpr |
              tl.tensor) = None,  # For blackwell, compute D = A @ B + D instead of D = A @ B. If None, default to True.
    pred=None,
    mBarriers: list[tlx.mbarrier] = [],
    two_ctas: bool = False,
    input_precision=None,
    out_dtype=tl.float32,
    _semantic=None,
) -> tl.tensor:
    """
    Performs a warp-group matrix multiply-accumulate operation of two blocks and return the matrix product.

    This maps directly to NVIDIA Hopper’s wgmma.mma_async instructions, enabling high-throughput matrix multiplication
    across multiple warps within a warpgroup, or Blackwell's tcgen05.mma instruction.

    The operation computes:
        D = A @ B + C

    Where:

        A: A matrix tile held in registers or shared memory

        B: A matrix tile loaded from shared memory

        C is an accumulator tile in registers

        D is the output tile in registers

    input_precision can be one of: tf32, tf32x3, ieee.
    """

    # Perform dot_precheck shared by tl.dot
    (A, B, acc_handle, input_precision, max_num_imprecise_acc,
     ret_ty) = (_semantic.dot_precheck(A, B, acc, input_precision, None, None, out_dtype, two_ctas))

    assert A.shape[0] >= 64, "M must be at least 64"
    assert A.shape[1] >= 16, "K must be at least 16"
    assert B.shape[1] >= 32, "N must be at least 32"

    cuda_compute_capability = int(cuda_parse_arch(_semantic.builder.options.arch))
    version = 5 if cuda_compute_capability >= 100 else 3

    # TODO. batched dot is not supported yet
    if isinstance(A, tlx.buffered_tensor) and A.type.storage == tlx.storage_kind.smem:
        A_handle = require_nv_mma_shared_layout(A, True, _semantic.builder)
    elif isinstance(A, tl.tensor):
        assert (cuda_compute_capability < 100), "register operand is not supported on Blackwell"
        A_handle = A.handle
    else:
        # A is in TMEM - create a dummy layout to be resolved after inlining
        # MMA operand A should have unpacked=False
        A_layout = tlx.DummyTMemLayoutEncoding(list(A.shape), A.dtype, unpacked=False)
        A_layout_handle = A_layout.to_ir(_semantic.builder)
        A_handle = _semantic.builder.create_require_layout(A.handle, A_layout_handle)

    B_handle = require_nv_mma_shared_layout(B, True, _semantic.builder)

    if version == 5:
        assert isinstance(A, tlx.buffered_tensor), "input must be a buffered tensor"
        # acc is in TMEM - create a dummy layout to be resolved after inlining
        # Accumulator D should have unpacked=True
        acc_layout = tlx.DummyTMemLayoutEncoding(list(acc.shape), acc.dtype, unpacked=True)
        acc_layout_handle = acc_layout.to_ir(_semantic.builder)
        acc_handle = _semantic.builder.create_require_layout(acc.handle, acc_layout_handle)
        handles = [t.handle for t in mBarriers]
        use_acc_handle = None
        if use_acc is not None:
            assert isinstance(use_acc, tl.tensor) or isinstance(
                use_acc, tl.constexpr), f"use_acc must be a tensor or constexpr, but got {type(use_acc)}"
            if isinstance(use_acc, tl.tensor):
                use_acc_handle = use_acc.handle
            else:
                use_acc_handle = _semantic.builder.get_int1(use_acc.value)
        output = _semantic.builder.create_tcgen5_dot(A_handle, B_handle, acc_handle, use_acc_handle, pred, two_ctas,
                                                     handles)
        return tl.tensor(output, tl.void)
    else:
        # Hopper path (version 3) - use dummy layouts that will be resolved after inlining

        # Create dummy MMA layout for accumulator
        mma_layout = tlx.DummyMMALayoutEncoding(list(acc.shape), acc.dtype,
                                                A.dtype,  # operand A element type for instruction shape calculation
                                                )
        mma_layout_handle = mma_layout.to_ir(_semantic.builder)
        acc = _semantic.builder.create_require_layout(acc_handle, mma_layout_handle)

        if isinstance(A, tl.tensor):
            # Create dummy dot operand layout for register operand A
            dot_op_layout = tlx.DummyDotOperandLayoutEncoding(list(A.shape), A.dtype, op_idx=0,  # 0 for operand A
                                                              )
            dot_op_layout_handle = dot_op_layout.to_ir(_semantic.builder)
            A_handle = _semantic.builder.create_require_layout(A.handle, dot_op_layout_handle)

        output = _semantic.builder.create_warp_group_dot(A_handle, B_handle, acc, input_precision,
                                                         max_num_imprecise_acc, True)
        # Release the mma layout for the output to conform to what the user expects
        output = _semantic.builder.create_release_layout(output)
        return tl.tensor(output, ret_ty)


@tl.builtin
def async_dot_scaled(
    A: tlx.buffered_tensor,
    B: tlx.buffered_tensor,
    acc: tlx.buffered_tensor,
    A_scale: tlx.buffered_tensor,
    A_format: str,
    B_scale: tlx.buffered_tensor,
    B_format: str,
    use_acc: (tl.constexpr |
              tl.tensor) = None,  # For blackwell, compute D = A @ B + D instead of D = A @ B. If None, default to True.
    pred=None,
    mBarriers: list[tlx.mbarrier] = [],
    out_dtype=tl.float32,
    _semantic=None,
) -> tl.tensor:
    """
    Performs a warp-group asynchronous scaled matrix multiply-accumulate (MMA)
    using Blackwell's `tcgen05.mma` instruction. This primitive is available only
    on NVIDIA Blackwell GPUs.

    The operation computed is:

        D = (A * A_scale) @ (B * B_scale) + D   (if use_acc is True)
        D = (A * A_scale) @ (B * B_scale)       (if use_acc is False)

    Inputs
    ------
    A : tlx.buffered_tensor
        Tile of matrix A, resident in shared memory (SMEM).

    B : tlx.buffered_tensor
        Tile of matrix B, resident in shared memory.

    acc : tlx.buffered_tensor
        Accumulator tile D, stored in tensor memory (TMEM). Used as both input
        and output when `use_acc=True`.

    A_scale : tlx.buffered_tensor
        Per-tile or per-subgroup scaling factors for operand A. Typically encoded
        as FP8 (E8M0) and stored in SMEM or TMEM.

    A_format : str
        FP8 format string for operand A (e.g., "e4m3", "e5m2"). Determines how
        the hardware interprets and scales FP8 inputs during MMA.

    B_scale : tlx.buffered_tensor
        Scaling factors for operand B, same semantics as A_scale.

    B_format : str
        FP8 format string for operand B.

    use_acc : tl.constexpr | tl.tensor, optional
        If True, performs an accumulate (D = A@B + D).
        If False, overwrites (D = A@B).
        If None, the default behavior is hardware-dependent (typically True).

    pred : optional
        Optional predicate masking for partial/conditional execution.

    mBarriers : list[tlx.mbarrier]
        Optional mbarriers used to coordinate producer/consumer warp-groups
        when `async_dot_scaled` participates in a pipelined MMA schedule.

    out_dtype : tl.dtype
        Output accumulation type before final store (default: fp32).

    Returns
    -------
    tl.tensor
        A TMEM tensor representing the updated accumulator tile D.
    """

    assert A.shape[0] >= 64, "M must be at least 64"
    assert A.shape[1] >= 16, "K must be at least 16"
    assert B.shape[1] >= 32, "N must be at least 32"

    cuda_compute_capability = int(cuda_parse_arch(_semantic.builder.options.arch))
    version = 5 if cuda_compute_capability >= 100 else 3
    assert version == 5, "async_dot_scaled is only available on Blackwell"

    assert isinstance(A, tlx.buffered_tensor), "input must be a buffered tensor"
    assert (A.type.storage == tlx.storage_kind.smem), "input must be a shared memory tensor"
    assert isinstance(B, tlx.buffered_tensor), "input must be a buffered tensor"
    assert (B.type.storage == tlx.storage_kind.smem), "input must be a shared memory tensor"

    # Require the shared memory layout for A and B
    A_handle = require_nv_mma_shared_layout(A, True, _semantic.builder)
    B_handle = require_nv_mma_shared_layout(B, True, _semantic.builder)

    # Handle input formats
    supported_formats = {"e2m1", "e4m3", "e5m2"}
    A_format = tl._unwrap_if_constexpr(A_format)
    B_format = tl._unwrap_if_constexpr(B_format)
    assert A_format in supported_formats, f"Unsupported A_format: {A_format}"
    assert B_format in supported_formats, f"Unsupported B_format: {B_format}"
    A_type = _semantic._str_to_fp_type(A_format)
    B_type = _semantic._str_to_fp_type(B_format)

    # Require the shared memory layout for A_scale and B_scale
    assert isinstance(A_scale, tlx.buffered_tensor), "A_scale must be a buffered tensor"
    assert (A_scale.type.storage == tlx.storage_kind.smem), "A_scale must be a shared memory tensor"
    assert isinstance(B_scale, tlx.buffered_tensor), "B_scale must be a buffered tensor"
    assert (B_scale.type.storage == tlx.storage_kind.smem), "B_scale must be a shared memory tensor"

    A_scale_handle = require_nv_mma_shared_layout(A_scale, False, _semantic.builder)
    B_scale_handle = require_nv_mma_shared_layout(B_scale, False, _semantic.builder)

    # acc is in TMEM - create a dummy layout to be resolved after inlining
    # Accumulator D should have unpacked=True
    acc_layout = tlx.DummyTMemLayoutEncoding(list(acc.shape), acc.dtype, unpacked=True)
    acc_layout_handle = acc_layout.to_ir(_semantic.builder)
    acc_handle = _semantic.builder.create_require_layout(acc.handle, acc_layout_handle)
    bar_handles = [t.handle for t in mBarriers]
    use_acc_handle = None
    if use_acc is not None:
        assert isinstance(use_acc, tl.tensor) or isinstance(
            use_acc, tl.constexpr), f"use_acc must be a tensor or constexpr, but got {type(use_acc)}"
        if isinstance(use_acc, tl.tensor):
            use_acc_handle = use_acc.handle
        else:
            use_acc_handle = _semantic.builder.get_int1(use_acc.value)
    output = _semantic.builder.create_tcgen5_dot_scaled(
        A_handle,
        B_handle,
        acc_handle,
        A_scale_handle,
        B_scale_handle,
        A_type,
        B_type,
        use_acc_handle,
        pred,
        bar_handles,
    )
    return tl.tensor(output, tl.void)


@tl.builtin
def async_dot_wait(
    pendings: tl.constexpr,
    inp: tl.tensor,
    _semantic=None,
) -> tl.tensor:
    """
    Wait for completion of prior asynchronous dot operations.
    Each input must be the tensors corresponding to the async dot ops that we're
    waiting on.
    """
    pendings = tl._unwrap_if_constexpr(pendings)
    return tl.tensor(
        _semantic.builder.create_warp_group_dot_wait([inp.handle], pendings)[0],
        inp.type,
    )


@tl.builtin
def tcgen05_commit(
    mBarrier: tlx.mbarrier,
    _semantic=None,
) -> tl.tensor:
    """
    Make the mbarrier track the completion of all prior asynchronous tcgen5 operations.
    NOTE: DO NOT use the same mBarrier passed to async_dot. This op needs a separate dedicated mBarrier.
    """
    return tl.tensor(_semantic.builder.create_tcgen05_commit(mBarrier.handle), tl.void)
