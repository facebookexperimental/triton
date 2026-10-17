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
    atomic_max
    atomic_min
    atomic_or
    atomic_poll
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

``triton.language.extra.tlx.amd_scaled_downcast_fp8(value)`` converts an FP32
tensor multiplied by 256 to E4M3FN on gfx950. The conversion instruction applies
the scale divisor 2^-8, so no separate FP32 multiply is needed. Rounding uses
round-to-nearest, ties-to-even and finite saturation. NaNs remain NaNs. The
operation preserves tensor ownership.


TLX Native Scaled MFMA
----------------------

.. autofunction:: triton.language.extra.tlx.amd_scaled_mfma_group

This gfx950 operation completes one native E4M3/E8M0 K64 fragment group.
Each participating wave must enable all 64 EXEC lanes.
Set full_wave=True only when the source guarantees this precondition.
Layouts do not establish full EXEC.
The compiler can reduce the entry delay from 20 states to four states.
All payload and scale inputs must pass its load or constant provenance checks.
Unknown producer graphs retain the 20-state entry delay.
Every group retains the 20-state result and payload drain.
The fused dependency drain does not replace memory waits or workgroup barriers.
