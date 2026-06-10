==========================================
Non-Power-of-2 (NPOT) Dimension Support
==========================================

Triton now supports non-power-of-2 tensor dimensions, gated behind the
:code:`TRITON_ALLOW_NPOT=1` environment variable. This lifts the constraint
that :code:`tl.arange` ranges and tensor shapes must be powers of 2.

.. note::

   This feature is experimental. Verify correctness and measure performance
   on your workload before adopting.

-----------------------
Enabling NPOT Support
-----------------------

Set the environment variable:

.. code-block:: bash

   TRITON_ALLOW_NPOT=1 python my_kernel.py

Or enable it programmatically:

.. code-block:: python

   import triton
   triton.knobs.language.allow_npot = True

Four enforcement gates verify that NPOT is enabled before allowing
non-power-of-2 shapes:

1. :code:`tl.arange` range check (``semantic.py``)
2. :code:`validate_block_shape` (``_utils.py``)
3. MLIR tensor size verification (``Traits.cpp``)
4. MLIR layout verification (``Dialect.cpp``)

-------------------------------
When NPOT Wins vs Pad-to-Pow2
-------------------------------

**Rule of thumb**: NPOT wins when padding to the next power of 2 wastes more
than ~20% of compute *and* the NPOT dimension has a large power-of-2 factor.

The power-of-2 factor matters because the coalescer extracts it for
:code:`sizePerThread`. Larger factors yield better memory coalescing.

.. list-table::
   :header-rows: 1

   * - Dimension
     - Next Pow2
     - Padding Waste
     - Pow2 Factor
     - Recommendation
   * - 48 (= 16 × 3)
     - 64
     - 25%
     - 16
     - **NPOT wins**
   * - 768 (= 256 × 3)
     - 1024
     - 25%
     - 256
     - **NPOT wins** (if using BLOCK_SIZE=768)
   * - 33 (= 1 × 33)
     - 64
     - 48%
     - 1
     - Marginal — benchmark
   * - 63 (= 1 × 63)
     - 64
     - 1.6%
     - 1
     - **Pad-to-pow2 wins**
   * - 100 (= 4 × 25)
     - 128
     - 22%
     - 4
     - Pad-to-pow2 likely better

When padding waste is below ~15%, the simpler XOR-based codegen from the
power-of-2 path is likely faster.

---------------------------------
Writing NPOT-Aware Kernels
---------------------------------

Use NPOT ranges directly in :code:`tl.arange`:

.. code-block:: python

   @triton.jit
   def add_kernel(X, Y, Out, SIZE: tl.constexpr):
       idx = tl.arange(0, SIZE)  # SIZE can be 48, 12, 7, etc.
       x = tl.load(X + idx)
       y = tl.load(Y + idx)
       tl.store(Out + idx, x + y)

Reductions work with NPOT block sizes:

.. code-block:: python

   @triton.jit
   def row_sum_kernel(X, Out, COLS: tl.constexpr):
       row_id = tl.program_id(0)
       cols = tl.arange(0, COLS)  # COLS can be NPOT
       x = tl.load(X + row_id * COLS + cols)
       tl.store(Out + row_id, tl.sum(x))

**Choosing block sizes**:

- If the problem dimension is small and NPOT (e.g., 48), set
  :code:`BLOCK_SIZE = problem_dim` to eliminate masking entirely.
- Prefer NPOT sizes with large pow2 factors: 48 (= 16 × 3) coalesces better
  than 7 (= 1 × 7).
- Masking still works normally with NPOT block sizes.

--------------------------
Validated Operators
--------------------------

NPOT block sizes have been validated on the following operator patterns:

- **Element-wise** (add, mul, etc.) — all hardware targets
- **Reductions** (``tl.sum``, ``tl.max``, etc.) — see :ref:`known-limitations`;
  multi-warp NPOT reductions are still being hardened on some targets
- **Softmax** — A100, MI350, GB200
- **Layer normalization** — A100, MI350, GB200

``tl.dot`` works with NPOT ``kWidth`` on NVIDIA (via ``modularIdentity1D``).
On AMD, MFMA requires M/N to be multiples of the MFMA atom dimensions;
see :ref:`known-limitations` for details.

NPOT support is validated on:

- **NVIDIA A100** (SM80)
- **NVIDIA GB200 / Blackwell** (SM100+)
- **AMD MI350** (CDNA4 / gfx950) — element-wise and reductions; ``tl.dot``
  with NPOT K works when K is a multiple of the MFMA kDim

--------------------------
Autotuner Integration
--------------------------

The autotuner supports NPOT block sizes through three APIs. All three require
:code:`TRITON_ALLOW_NPOT=1` (or :code:`triton.knobs.language.allow_npot = True`);
when NPOT is not enabled, they are no-ops that return the original configs or an
empty list.

**include_npot on @triton.autotune**

The simplest option: set :code:`include_npot=True` on the :code:`@triton.autotune`
decorator. For every config that has an NPOT block size, the autotuner
automatically adds a pow2-padded variant so the two can be compared head-to-head.

.. code-block:: python

   @triton.autotune(
       configs=[
           triton.Config({"BLOCK_M": 48, "BLOCK_K": 64}),
           triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}),
       ],
       key=["M", "K"],
       include_npot=True,  # adds a BLOCK_M=64 padded variant for the 48-config
   )
   @triton.jit
   def my_kernel(...):
       ...

**triton.npot_block_sizes(min_val, max_val, base_multiple=32)**

Returns a list of NPOT block sizes that are multiples of :code:`base_multiple`
in the range :code:`[min_val, max_val]`. Use this to generate NPOT candidates
for your config list.

.. code-block:: python

   # Returns [96, 160, 192, 224] — multiples of 32 that are not powers of 2
   npot_sizes = triton.npot_block_sizes(32, 256, base_multiple=32)

   configs = [triton.Config({"BLOCK_M": s}) for s in npot_sizes]

**triton.expand_configs_npot(configs, block_size_keys=None)**

For each config with an NPOT block size, adds a pow2-padded variant for A/B
comparison. This is the function that :code:`include_npot=True` calls internally.

By default, it detects block-size keys by looking for kwargs whose name contains
``BLOCK`` (case-insensitive). Pass :code:`block_size_keys` explicitly to override.

.. code-block:: python

   configs = [
       triton.Config({"BLOCK_M": 48, "BLOCK_N": 128}),
       triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}),
   ]
   # Adds a variant with BLOCK_M=64 (next pow2 of 48) for the first config
   expanded = triton.expand_configs_npot(configs)

.. note::

   Under the hood, NPOT block sizes use Barrett reduction for index
   computation, replacing expensive integer remainder (UREM) with a
   multiply-shift sequence. This narrows the performance gap vs pow2 block
   sizes significantly.

--------------------------
Performance Expectations
--------------------------

.. list-table::
   :header-rows: 1

   * - Aspect
     - Pow2 (XOR) Path
     - NPOT (ADD + Barrett) Path
   * - Index computation
     - Single XOR per basis vector
     - ADD + select per basis, Barrett modulo (mul+shift, ~4–6 cycles vs ~20–40 for UREM)
   * - Index codegen
     - XOR per basis vector with diagonal extraction
     - Linear ADD chain, single Barrett modulo at end
   * - sizePerThread
     - Uses full contiguity
     - Uses largest pow2 factor of contiguity (via ``highestPowOf2Divisor``)
   * - Memory coalescing
     - Optimal
     - Good when pow2 factor is large

**Barrett reduction** replaces hardware integer remainder (UREM) with a
multiply-shift sequence, costing ~4–6 cycles vs ~20–40 for UREM.
The modular path in ``matrixVectorProd`` accumulates index contributions via
a linear ADD chain without intermediate modulo; ``applyNpotModulo`` applies
a single Barrett reduction at the end.

**Identity basis fast-path**: when a modular layout has identity bases
``[1, 2, 4, ...]`` (the common case for stride-1 dimensions), the entire
select+add loop collapses to a single AND mask. For example, a dimension of
size 127 drops from 28 select+add ops to 1 AND.

**smallModulo optimization**: when the compiler can prove the accumulated
value is less than ``2 * outDimSize`` (i.e., the sum of all basis entries is
small enough), it uses a compare-and-subtract (2 ops) instead of Barrett
reduction (6 ops). This fires frequently on AMD gfx950 where i64 multiply
is slow.

**sizePerThread alignment clamping** extracts the largest power-of-2
factor of non-pow2 contiguity via ``highestPowOf2Divisor`` (e.g.,
contiguity 48 yields sizePerThread 16, not 32), ensuring valid vector
load widths even with NPOT contiguity.

-------------------------------
Benchmark Summary
-------------------------------

Results from softmax and layer normalization benchmarks across three
hardware targets (10 NPOT shapes each, comparing NPOT block sizes vs
pad-to-pow2):

.. list-table::
   :header-rows: 1

   * - Hardware
     - Softmax
     - Layer Norm
   * - A100 (SM80)
     - Mostly competitive (2–7% overhead), some shapes match or beat pow2
     - NPOT faster across all shapes (2–72%)
   * - GB200 (SM100)
     - Up to 45% faster at large dims (5120, 6144)
     - Up to 44% faster at large dims
   * - MI350 (CDNA4)
     - Modest 3–6% overhead
     - 60–62% faster at N=1023/2047

**Key takeaways**:

- Layer normalization benefits most because it is memory-bound and NPOT
  eliminates wasted bandwidth from padding.
- Large NPOT dimensions (768+) with high pow2 factors show the biggest
  gains.
- At boundary sizes (e.g., 1023, 2047), NPOT avoids crossing a
  power-of-2 rep boundary, sometimes yielding dramatic speedups.

-------------------------------
Hardware Support Matrix
-------------------------------

.. list-table::
   :header-rows: 1

   * - Feature
     - A100 (SM80)
     - MI350 (CDNA4)
     - Blackwell (SM100+)
   * - Element-wise + reductions
     - Yes
     - Yes (``warpsPerTile`` fix, ``divideCeil``)
     - Yes
   * - ``tl.dot`` with NPOT kWidth
     - Yes (``modularIdentity1D``)
     - Partial — MFMA M/N must be multiples of mDim/nDim
     - Yes (layout algebra ready)
   * - Shared memory NPOT tiles
     - Yes (``getCoreMatrixLinearLayout`` fix)
     - Standard shared: yes. Rotating: no (XOR-only)
     - Yes
   * - Barrett / smallModulo codegen
     - Yes
     - Yes (``smallModulo`` preferred — avoids slow i64 mul)
     - Yes

.. _known-limitations:

-------------------------------
Known Limitations
-------------------------------

AMD MFMA with NPOT M/N
   ``chooseMfmaInstruction()`` returns ``failure()`` when
   ``M % mDim != 0 || N % nDim != 0``, preventing ``tl.dot`` with NPOT
   M or N dimensions on AMD. Element-wise ops and reductions work via the
   ``warpsPerTile`` NPOT fix (``divideCeil`` + pow2-rounding for warp
   distribution). NPOT K dimension works when K is a multiple of the MFMA
   kDim.

AMD rotating shared memory
   ``sharedToLinearLayoutAMDRotating`` uses XOR-based bank conflict
   avoidance that assumes pow2 shapes. Standard (non-rotating) shared
   memory layouts work with NPOT shapes.

warpReduce with NPOT lane counts
   NPOT reductions round up to the next power of 2 and predicate
   out-of-range lanes with the identity element. If the combine region
   uses an unsupported op (not add/mul/max/min/and/or/xor), compilation
   fails with "cannot determine identity element". Use standard reduction
   ops to avoid this.

``invertAndCompose`` multi-rep NPOT
   When multi-rep NPOT configurations produce coefficients > 1, the
   ``lstsqModular`` solver handles CRT-based resolution.
   Validated on A100, GB200, and MI350 for shapes up to 6144.

SM100 FP32 reduction precision
   NPOT reductions on SM100 (GB200) may show slightly higher FP32
   rounding error than A100 due to different accumulation order from
   wrapping predicates. Results pass at atol=1e-4; the pow2 control
   shows similar variance.
