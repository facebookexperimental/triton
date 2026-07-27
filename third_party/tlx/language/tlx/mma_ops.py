import triton.language.core as tl

from . import types as tlx
from .utility import cuda_parse_arch


@tl.builtin
def require_layout(x, layout, _semantic=None):
    """Pin a register tensor's layout as a hard user anchor.

    ``layout`` is a ``tlx.layout(...)`` (Shape:Stride) mapped to a ``#linear``
    encoding and wrapped as ``#tlx.user_layout`` (PinnedEncodingTrait), so the
    downstream layout passes (tritongpu-coalesce, remove-layout-conversions, AMD
    optimize-epilogue) treat it as fixed and never rewrite it. Unlike a plain
    (no_verify) require_layout, this survives to Coalesce and lets you pin an
    epilogue ``tl.store`` to a coalesced register layout *without* staging the
    value through LDS.
    """
    layout = tl._unwrap_if_constexpr(layout)
    enc = layout.to_ir(_semantic.builder, x.shape, x.dtype)
    handle = _semantic.builder.create_require_layout(x.handle, enc, pin=True)
    return tl.tensor(handle, x.type)


def require_nv_mma_shared_layout(x: tlx.buffered_tensor, swizzled: bool, _builder=None, fp4Padded: bool = False):
    assert isinstance(x.type.layout, tlx.shared_layout_encoding), "input must be a shared tensor"
    rank = len(x.shape)
    layout = tlx.nv_mma_shared_layout_encoding(
        shape=x.shape,
        order=x.type.layout.order,
        elemType=x.dtype,
        numCTAsPerCGA=[1] * rank,
        numCTASplit=[1] * rank,
        numCTAOrder=list(reversed(range(rank))),
        fp4Padded=fp4Padded,
        swizzled=swizzled,
    )

    layout_handle = _builder.make_nv_mma_shared_encoding_attr(
        [int(x) for x in layout.shape],
        layout.order,
        layout.elemType.to_ir(_builder),
        layout.numCTAsPerCGA,
        layout.numCTASplit,
        layout.numCTAOrder,
        layout.fp4Padded,
        layout.swizzled,
    )
    return _builder.create_require_layout(x.handle, layout_handle)


def require_dot_operand_layout(opnd: tl.tensor, opIdx, parent_layout, _builder=None):
    layout_handle = _builder.make_dot_operand_encoding_attr(opnd.handle, opIdx, parent_layout)
    return _builder.create_require_layout(opnd.handle, layout_handle)


def require_tmem_layout(src: tlx.buffered_tensor, col_stride: int, cta_mode: int = tlx.TMemCTAMode.DEFAULT,
                        _builder=None):
    assert (isinstance(src, tlx.buffered_tensor) and src.type.storage == tlx.storage_kind.tmem
            and isinstance(src.type.layout, tlx.tensor_memory_layout_encoding)), "input must be a TMEM tensor"
    old_layout = src.type.layout
    if old_layout.colStride != col_stride or old_layout.ctaMode != cta_mode:
        layout_handle = _builder.make_tensor_memory_encoding_attr(
            old_layout.blockM,
            old_layout.blockN,
            col_stride,
            old_layout.CTASplitM,
            old_layout.CTASplitN,
            cta_mode,
        )
        return _builder.create_require_layout(src.handle, layout_handle)
    # if the layout is already correct, return the original handle
    return src.handle


def require_tmem_scales_layout(src: tlx.buffered_tensor, _builder=None):
    """
    Require tensor memory scales layout for a TMEM tensor.
    """
    assert isinstance(
        src, tlx.buffered_tensor) and src.type.storage == tlx.storage_kind.tmem, ("input must be a TMEM tensor")
    layout = tlx.tensor_memory_scales_layout_encoding.make_default()
    layout_handle = layout.to_ir(_builder)
    return _builder.create_require_layout(src.handle, layout_handle)


def _get_use_acc_handle(use_acc: tl.constexpr | tl.tensor | None, _builder):
    if use_acc is None:
        return None
    assert isinstance(use_acc, tl.tensor) or isinstance(
        use_acc, tl.constexpr), (f"use_acc must be a tensor or constexpr, but got {type(use_acc)}")
    if isinstance(use_acc, tl.tensor):
        return use_acc.handle
    return _builder.get_int1(use_acc.value)


# async dot signature needs to be close to tl.dot as much as possible
@tl.builtin
def async_dot(
    A: tlx.buffered_tensor | tl.tensor,
    B: tlx.buffered_tensor,
    acc: tlx.buffered_tensor | tl.tensor | None = None,
    use_acc: tl.constexpr
    | tl.tensor = None,  # For blackwell, compute D = A @ B + D instead of D = A @ B. If None, default to True.
    pred=None,
    mBarriers: list[tlx.mbarrier] = [],
    two_ctas: bool = False,
    force_async: bool = False,
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
     ret_ty) = _semantic.dot_precheck(A, B, acc, input_precision, None, None, out_dtype, two_ctas)

    cuda_compute_capability = int(cuda_parse_arch(_semantic.builder.options.arch))
    version = 5 if cuda_compute_capability >= 100 else 3

    M, K, N = A.shape[0], A.shape[1], B.shape[1]
    # K is only lower-bounded: the lowering splits the K blocks across instructions,
    # so there is no maximum to enforce here.
    assert K >= 16, "K must be at least 16"
    if version == 5:
        # Blackwell tcgen05.mma (PTX ISA 9.3, Table 42): the per-CTA instruction shape
        # is M x N with M in {64, 128} and N a multiple of 8 in [8, 256]. With CTA
        # group 2 the logical M/N double across the two CTAs, but the per-CTA operand
        # tiles passed here still follow this rule.
        assert M in (64, 128), f"M must be 64 or 128 on Blackwell, but got {M}"
        assert 8 <= N <= 256 and N % 8 == 0, f"N must be a multiple of 8 in [8, 256], but got {N}"
    else:
        # Hopper wgmma.mma_async (PTX ISA 9.3): M is a multiple of 64 and N a multiple
        # of 8 in [8, 256].
        assert M % 64 == 0, f"M must be a multiple of 64 on Hopper, but got {M}"
        assert 8 <= N <= 256 and N % 8 == 0, f"N must be a multiple of 8 in [8, 256], but got {N}"

    # TODO. batched dot is not supported yet
    a_is_tmem = isinstance(A, tlx.buffered_tensor) and A.type.storage == tlx.storage_kind.tmem
    a_cta_mode = tlx.TMemCTAMode.DEFAULT
    acc_cta_mode = tlx.TMemCTAMode.DEFAULT
    if two_ctas and isinstance(acc, tlx.buffered_tensor) and acc.type.layout.blockM == 64:
        acc_cta_mode = tlx.TMemCTAMode.TwoCTA_RHS
        if a_is_tmem:
            a_cta_mode = tlx.TMemCTAMode.TwoCTA_LHS

    if isinstance(A, tlx.buffered_tensor) and A.type.storage == tlx.storage_kind.smem:
        A_handle = require_nv_mma_shared_layout(A, True, _semantic.builder)
    elif isinstance(A, tl.tensor):
        assert cuda_compute_capability < 100, "register operand is not supported on Blackwell"
        A_handle = A.handle
    else:
        # set colStride to 1 (packed) for A, and set cta_mode
        A_handle = require_tmem_layout(A, 1, a_cta_mode, _semantic.builder)

    B_handle = require_nv_mma_shared_layout(B, True, _semantic.builder)

    if version == 5:
        assert isinstance(A, tlx.buffered_tensor), "input must be a buffered tensor"
        # D needs colStride = 32 / bitwidth, see https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-packing-formats
        acc_handle = require_tmem_layout(acc, 1, acc_cta_mode, _semantic.builder)
        handles = [t.handle for t in mBarriers]
        is_async = force_async or len(handles) > 0
        use_acc_handle = _get_use_acc_handle(use_acc, _semantic.builder)
        output = _semantic.builder.create_tcgen5_dot(A_handle, B_handle, acc_handle, use_acc_handle, pred, two_ctas,
                                                     handles, is_async)
        return tl.tensor(output, tl.void)
    else:
        mma_layout = _semantic.builder.make_nv_mma_encoding_attr(A_handle, acc_handle, version, 0,
                                                                 _semantic.builder.options.num_warps)
        acc = _semantic.builder.create_require_layout(acc_handle, mma_layout)
        if isinstance(A, tl.tensor):
            A_handle = require_dot_operand_layout(A, 0, mma_layout, _semantic.builder)
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
    use_acc: tl.constexpr
    | tl.tensor = None,  # For blackwell, compute D = A @ B + D instead of D = A @ B. If None, default to True.
    pred=None,
    mBarriers: list[tlx.mbarrier] = [],
    two_ctas: bool = False,
    force_async: bool = False,
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
        as FP8 (E8M0) and stored in SMEM or TMEM. The storage type is automatically
        detected from the tensor's storage attribute.

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

    two_ctas : bool
        If True, the op will execute a matmul across two contiguous CTAs,
        reading data distributed across the two CTAs. Default is False.

    out_dtype : tl.dtype
        Output accumulation type before final store (default: fp32).

    Returns
    -------
    tl.tensor
        A TMEM tensor representing the updated accumulator tile D.
    """

    cuda_compute_capability = int(cuda_parse_arch(_semantic.builder.options.arch))
    version = 5 if cuda_compute_capability >= 100 else 3
    assert version == 5, "async_dot_scaled is only available on Blackwell"

    M, K, N = A.shape[0], A.shape[1], B.shape[1]
    # Blackwell block-scaled tcgen05.mma (PTX ISA 9.3, Table 42): the per-CTA
    # instruction shape is fixed to M=128 with N a multiple of 8 in [8, 256]. K is
    # only lower-bounded because the lowering splits the K blocks across instructions.
    # Gap: the ISA allows N up to 256, but the current lowering rejects blockN==256 for
    # the scaled op (see InsertTmemAref.cpp), so we cap N at 128 until that is fixed.
    assert M == 128, f"M must be 128 for the scaled MMA, but got {M}"
    assert K >= 16, "K must be at least 16"
    assert 8 <= N <= 128 and N % 8 == 0, f"N must be a multiple of 8 in [8, 128], but got {N}"

    assert isinstance(A, tlx.buffered_tensor), "input must be a buffered tensor"
    assert isinstance(B, tlx.buffered_tensor), "input must be a buffered tensor"
    assert B.type.storage == tlx.storage_kind.smem, "input must be a shared memory tensor"

    # Handle input formats
    supported_formats = {"e2m1", "e4m3", "e5m2"}
    A_format = tl._unwrap_if_constexpr(A_format)
    B_format = tl._unwrap_if_constexpr(B_format)
    assert A_format in supported_formats, f"Unsupported A_format: {A_format}"
    assert B_format in supported_formats, f"Unsupported B_format: {B_format}"
    A_type = _semantic._str_to_fp_type(A_format)
    B_type = _semantic._str_to_fp_type(B_format)

    a_is_tmem = A.type.storage == tlx.storage_kind.tmem
    a_cta_mode = tlx.TMemCTAMode.DEFAULT
    acc_cta_mode = tlx.TMemCTAMode.DEFAULT
    if two_ctas and acc.type.layout.blockM == 64:
        acc_cta_mode = tlx.TMemCTAMode.TwoCTA_RHS
        if a_is_tmem:
            a_cta_mode = tlx.TMemCTAMode.TwoCTA_LHS

    # Require layout for A: SMEM or TMEM (mirroring async_dot's 3-way branch)
    is_A_fp4 = A_format == "e2m1"
    is_B_fp4 = B_format == "e2m1"
    is_mixed_precision = A_format != B_format
    if A.type.storage == tlx.storage_kind.smem:
        A_fp4Padded = is_A_fp4 and is_mixed_precision
        A_handle = require_nv_mma_shared_layout(A, True, _semantic.builder, fp4Padded=A_fp4Padded)
    else:
        assert a_is_tmem, "A must be in SMEM or TMEM"
        A_handle = require_tmem_layout(A, 1, a_cta_mode, _semantic.builder)

    # Require layout for B (always SMEM)
    B_fp4Padded = is_B_fp4 and is_mixed_precision
    B_handle = require_nv_mma_shared_layout(B, True, _semantic.builder, fp4Padded=B_fp4Padded)

    # Handle scale tensors - can be in SMEM or TMEM (auto-detected from storage type)
    assert isinstance(A_scale, tlx.buffered_tensor), "A_scale must be a buffered tensor"
    assert isinstance(B_scale, tlx.buffered_tensor), "B_scale must be a buffered tensor"

    if A_scale.type.storage == tlx.storage_kind.tmem:
        A_scale_handle = require_tmem_scales_layout(A_scale, _semantic.builder)
    else:
        assert A_scale.type.storage == tlx.storage_kind.smem, "A_scale must be in SMEM or TMEM"
        A_scale_handle = require_nv_mma_shared_layout(A_scale, False, _semantic.builder)

    if B_scale.type.storage == tlx.storage_kind.tmem:
        B_scale_handle = require_tmem_scales_layout(B_scale, _semantic.builder)
    else:
        assert B_scale.type.storage == tlx.storage_kind.smem, "B_scale must be in SMEM or TMEM"
        B_scale_handle = require_nv_mma_shared_layout(B_scale, False, _semantic.builder)

    # D needs colStride = 32 / bitwidth, see https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-packing-formats
    acc_handle = require_tmem_layout(acc, 1, acc_cta_mode, _semantic.builder)
    bar_handles = [t.handle for t in mBarriers]
    is_async = force_async or len(bar_handles) > 0
    use_acc_handle = _get_use_acc_handle(use_acc, _semantic.builder)
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
        two_ctas,
        bar_handles,
        is_async,
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
    return tl.tensor(_semantic.builder.create_warp_group_dot_wait([inp.handle], pendings)[0], inp.type)


@tl.builtin
def tcgen05_commit(
    mBarrier: tlx.mbarrier,
    two_ctas: bool = False,
    _semantic=None,
) -> tl.tensor:
    """
    Make the mbarrier track the completion of all prior asynchronous tcgen5 operations.
    NOTE: DO NOT use the same mBarrier passed to async_dot. This op needs a separate dedicated mBarrier.
    """
    if not two_ctas:
        pred_handle = _semantic.builder.get_int1(True)
    else:
        # cluster_cta_rank() % 2 == 0
        cta_rank = _semantic.builder.create_cluster_cta_rank()
        mod_result = _semantic.builder.create_urem(cta_rank, _semantic.builder.get_int32(2))
        pred_handle = _semantic.builder.create_icmpEQ(mod_result, _semantic.builder.get_int32(0))
    return tl.tensor(_semantic.builder.create_tcgen05_commit(mBarrier.handle, pred_handle), tl.void)
