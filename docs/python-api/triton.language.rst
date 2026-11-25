triton.language
===============

.. currentmodule:: triton.language


Programming Model
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    tensor
    tensor_descriptor
    program_id
    num_programs
    map_elementwise


Creation Ops
------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    arange
    cat
    full
    zeros
    zeros_like
    cast
    to_tensor


Shape Manipulation Ops
----------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    broadcast
    broadcast_to
    expand_dims
    interleave
    join
    permute
    ravel
    reshape
    split
    squeeze
    trans
    unsqueeze
    view


Linear Algebra Ops
------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    dot
    dot_scaled


Memory/Pointer Ops
------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    load
    store
    make_tensor_descriptor
    load_tensor_descriptor
    store_tensor_descriptor
    make_block_ptr
    advance


Indexing Ops
------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    flip
    where
    swizzle2d


Math Ops
--------

.. autosummary::
    :toctree: generated
    :nosignatures:

    abs
    add
    cdiv
    ceil
    clamp
    cos
    div_rn
    erf
    exp
    exp2
    fdiv
    floor
    fma
    log
    log2
    maximum
    minimum
    mul
    rsqrt
    sigmoid
    sin
    softmax
    sqrt
    sqrt_rn
    sub
    umulhi


Reduction Ops
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    all
    any
    argmax
    argmin
    max
    min
    reduce
    reduce_or
    sum
    xor_sum

Scan/Sort Ops
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    associative_scan
    cumprod
    cumsum
    histogram
    sort
    topk
    gather

Atomic Ops
----------

.. autosummary::
    :toctree: generated
    :nosignatures:

    atomic_add
    atomic_and
    atomic_cas
    atomic_load
    atomic_max
    atomic_min
    atomic_or
    atomic_poll
    atomic_store
    atomic_xchg
    atomic_xor

Random Number Generation
------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    randint4x
    randint
    rand
    rand4x
    randn
    randn4x


Iterators
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    range
    static_range


Inline Assembly
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    inline_asm_elementwise


Compiler Hint Ops
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    assume
    debug_barrier
    max_constancy
    max_contiguous
    multiple_of


Debug Ops
-----------------

.. autosummary::
    :toctree: generated
    :nosignatures:

    static_print
    static_assert
    device_print
    device_assert
    expect_zero


TLX AMD Conversion
------------------

``triton.language.extra.tlx.amd_scaled_downcast_fp8(value, scale, dtype=tl.float8e4nv)``
converts ``value * scale`` from FP32 to E4M3FN or E5M2 on gfx950. ``scale`` is a
compile-time power of two from 2^-127 to 2^126; it becomes the constant E8M0
divisor of ``amdg.scaled_downcast_fp8`` (``scale_const = 127 - log2(scale)``), so
the conversion instruction applies the factor and no separate FP32 multiply is
needed. Rounding is round-to-nearest-even; NaNs and values beyond the FP8 range
convert to NaN. The operation preserves tensor ownership.

``triton.language.extra.tlx.amd_scaled_upcast_fp4(x, scale, dtype=tl.bfloat16, axis=-1)``
converts packed E2M1 (FP4) values times their E8M0 scales to BF16 or F16 on gfx950
and gfx1250. ``x`` is a uint8 tensor with two E2M1 values per byte along ``axis``
(the even element in the low nibble); the result doubles that extent. ``scale``
holds raw E8M0 bytes (value ``2^(byte - 127)``) with the result's shape except along
``axis``, where each scale covers a multiple of 8 consecutive values (32 for MXFP4).
The conversion is exact. It lowers to ``amdg.scaled_upcast_fp4``, which applies the
scale in the conversion instruction (``v_cvt_scale_pk8_*_fp4`` on gfx1250,
``v_cvt_scalef32_pk_*_fp4`` on gfx950).

``triton.language.extra.tlx.amd_scaled_downcast_fp4(x, scale, axis=-1)`` divides
FP32, BF16 or F16 values by their E8M0 scales and packs them to E2M1 on gfx950 and
gfx1250, returning a uint8 tensor with two values per byte along ``axis``. Each
scale covers a multiple of 8 consecutive values along ``axis``. E2M1 rounding is to
nearest even, saturating at 6. It lowers to ``amdg.scaled_downcast_fp4``
(``v_cvt_scalef32_pk8_fp4_*`` on gfx1250, ``v_cvt_scalef32_pk_fp4_*`` on gfx950).
