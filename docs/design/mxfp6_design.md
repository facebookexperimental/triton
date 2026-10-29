# MXFP6 (e2m3/e3m2) data type support in Triton and TLX

**Scope:** OCP microscaling FP6 formats (`e2m3`, `e3m2`) in `tl.dot_scaled`,
Gluon scaled-MMA entry points, and TLX `async_dot_scaled`, on NVIDIA sm100+
and AMD gfx950/gfx1250.

This document describes the design for FP6 dtype support: where it is
exposed, how packed data is laid out in memory, how it flows through loads,
stores, layouts, CUDA-core (de)quantization, shared memory/LDS, scaled dots,
and TMEM. It distinguishes what is implemented today from what the lowering
is designed to support.

Background on the microscaling formats themselves: the OCP MX specification
(`https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf`).
FP6 has no Inf/NaN; subnormals are supported. The two encodings are:

| Format | Sign | Exponent bits / bias | Mantissa bits |
|---|---|---|---|
| `e2m3` | 1 | 2 / 1 | 3 |
| `e3m2` | 1 | 3 / 3 | 2 |

## 1. Where FP6 is supported

### Triton (`tl.dot_scaled`)

`tl.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, ...)`
accepts `lhs_format`/`rhs_format` in
`{e2m1, e2m3, e3m2, e4m3, e5m2, bf16, fp16}`. FP6 operands must be K-packed:
packed as `uint8` along the K dimension with `lhs_k_pack`/`rhs_k_pack` true
(the default); non-K-packed FP6 is rejected. The frontend builds
`tt.dot_scaled` with `ScaleDotElemType::E2M3`/`E3M2`, and the IR verifier
enforces the packing contract: with K-packing, a logical K dimension that is
not a multiple of 16 is an error (16 FP6 values form one 96-bit /
12-byte dense group), and on the NVIDIA side the storage element type
must be `i8`. Scale metadata follows the existing MX convention (e8m0
scales, one per 32 values along K) with FP6-aware scale-factor deduction.

The Triton interpreter (`TRITON_INTERPRET=1`) unpacks and decodes FP6, so
numerics are testable without hardware.

### Gluon

- AMD CDNA4 `mfma_scaled`: `e2m3`/`e3m2` accepted for both operands.
- AMD CDNA5 `wmma_scaled`: `e2m3`/`e3m2` accepted; FP6 operands require WMMA
  instruction shape `[16, 16, 96]`, and the valid dtype combinations include
  FP6 for both sides.
- NVIDIA Blackwell `tcgen05_mma_scaled`: `e2m3`/`e3m2` accepted for both
  operand types.
- Shared AMD helpers treat FP6 K as already unpacked (padded) and assert
  divisibility by 4 when computing scale shapes.

### TLX

`tlx.async_dot_scaled` accepts `A_format`/`B_format` in
`{e2m1, e2m3, e3m2, e4m3, e5m2}`. It is Blackwell-only (compute capability
>= 10.0).

### Hardware coverage

Numerics and codegen are tested on NVIDIA sm100+ (with M >= 128 for the
tcgen05 path), AMD gfx950 (MFMA path), and AMD gfx1250 (WMMA path). Today all
three lower FP6 scaled dots through the decompose/upcast path (section 8):
generated code contains plain `tc_gen5_mma` / `v_mfma_f32_` / `v_wmma_f32_`
and never native `mxf8f6f4` / `f8f6f4` instructions.

## 2. Representation in memory (HBM)

There is no 6-bit hardware dtype anywhere in the pipeline. An FP6 tensor is
physically a `uint8` tensor; the 6-bit values are dense LSB-first packed
codes inside the byte stream, packed along the K axis (last dim for LHS
`[M, K]`, dim 0 for RHS `[K, N]`):

- Within each 16-value group, every 4 consecutive logical values
  `e0..e3` (24 bits) occupy 3 bytes:
  `b0 = e0 | e1[1:0]<<6`, `b1 = e1[5:2] | e2[3:0]<<4`,
  `b2 = e2[5:4] | e3<<2`. Code `i` lives in bits `[6*i, 6*i+6)` of the
  byte stream.
- K logical elements therefore produce `3K/4` bytes of real packed data
  (16 values per 12-byte group).
  Host storage (`MXFP6Tensor`) is exactly that dense stream: a logical
  `[M, K]` FP6 matrix is a physical `[M, 3K/4]` uint8 tensor with no
  padding byte. This follows the group-of-16 granularity suggested
  upstream: 16 FP6 values occupy 96 bits (12 bytes, three 32-bit words),
  and gfx950 consumes 32 values (two such groups) per step.
- Some kernels expand the dense stream into a logical-width register /
  shared-memory tile on load (leading `3K/4` slots hold bytes, trailing
  quarter masked off), and the Blackwell TMA path inserts padding bytes
  in shared memory for the `mxf8f6f4` container format. Those are
  on-chip/load-time representations of the same dense global stream,
  not the host storage format.

Consequences of the dense layout:

- The tensor-level divisibility step is K % 16: 16 values per 96-bit
  group. This is asserted in the frontend, the IR verifiers, and the
  interpreter. Host packing alone only needs groups of 4 values
  (3 bytes); the % 16 rule is the scaled-dot contract layered on top.
- In `MXFP6Tensor` storage no bytes are padding: dense storage saves
  25% of HBM traffic and footprint versus a padded 4-values-per-4-bytes
  layout. The Blackwell TLX operator separately accepts logical-width
  input tensors whose trailing quarter is ignored padding.
- The 3-byte groups do not align to 16-byte lines (lcm(3,16)=48), so group
  boundaries drift within any 16-byte chunk. Unpacking code must assemble
  values from bytes spanning line boundaries; this is purely a consumer-side
  (register/smem) concern once alignment-safe movement is guaranteed.
- Finer % 4 steps appear only inside lowering, where offsets/extents
  within an already-validated % 16 tile step by 4 values (3 bytes); they
  are not a tensor-level contract and do not admit K % 4 tensors.

Scales are a separate plain uint8 tensor (e8m0, one scale per 32 K-elements)
with no packing.

## 3. Loads: `tl.load`, TMA/TDM, async loads and pipelining

### Regular `tl.load`

A `tl.load` of an FP6 operand is an ordinary blocked load of the physical
uint8 tensor. Loads are dtype-agnostic: they move bytes, and FP6 unpacking /
dequantization happens afterwards in registers (section 6) or after the
shared-memory stage. No FP6-specific load op exists, and masking/other
semantics are unchanged from uint8 loads.

### TMA (NVIDIA)

There is no 6-bit TMA descriptor element type, and no FP6-specific TMA
support. By design, FP6 tiles move through TMA as UINT8 tensors. The
Blackwell native path additionally uses the padded shared-memory
container of section 2 (`.b6x16_p32`), with one extra constraint. The
descriptors formed in this stack today are over logical-width tensors
(physical `[M, K]` uint8, dense stream in the leading `3K/4` of each
row, as the Blackwell TLX operator takes), so their row stride is
K bytes:

- Tensor-map base address must be 16-byte aligned. This is free for normally
  allocated tensors, and every row starts 16-byte aligned as long as the row
  stride (K bytes) is a multiple of 16.
- Non-inner strides must be multiples of 16 bytes, so a 2D TMA descriptor
  over a logical-width FP6 tensor requires **K a multiple of 16** -- the
  same step as the tensor-level contract of section 2, not a stricter
  rule. Describing the dense `[M, 3K/4]` `MXFP6Tensor` stream directly
  instead makes the row stride `3K/4` bytes, which is a multiple of 16
  only when **K is a multiple of 64**; for K a multiple of 16 but not
  64, 1D bulk copies or `cp.async` remain the fallback for the dense
  stream.
- Inner box dims must be multiples of 16 bytes.
- Swizzle modes (32B/64B/128B) apply on the shared-memory side purely in byte
  terms, so standard swizzle selection works unchanged.
- TMA completion counts the packed global bytes (`3/4` of the logical
  width); for the native path, padding is inserted in the shared-memory
  container. It is not part of `MXFP6Tensor` storage.

See `design/triton_tma_multicast.md` for TMA background; the same descriptor
mechanics apply, with uint8 as the element type.

### TDM (AMD)

TDM is AMD's descriptor-based async copy engine (tensor descriptors with
`async_load`/`async_store`, the counterpart to TMA), enabled on gfx950 and
gfx1250. As with TMA, FP6 has no descriptor-level type: tiles move as packed
uint8 bytes through the existing descriptor load path, and unpacking happens
downstream. The standing TDM guidance applies unchanged -- in particular,
keeping the inner-most block dim at 256 bytes for good TDM performance,
which in the logical-width load layout is 256 logical K elements.

### Async loads and pipelining

Descriptor loads (TMA/TDM) participate in loop pipelining normally:
multi-stage buffers, mbarriers/async-wait counts, prologue/epilogue overlap.
FP6 introduces no new pipelining mechanics: the operand that is staged and
pipelined is the uint8 byte tensor (plus the small uint8 scale tensor), and
the elementwise unpack runs after the async wait, exactly where any input
elementwise op would run. Persistent-kernel / warp-specialized schedules
treat FP6 inputs as uint8 inputs with a fused dequant epilogue on the load.

## 4. Stores: `tl.store`, TMA/TDM, and SWP

There are no packed-FP6 store operations, and none are needed for the
supported flows: `dot_scaled` outputs are compute type (f32 accumulator,
typically stored as f32/bf16/f16), so every store on the output path --
`tl.store`, TMA stores, TDM `async_store` -- is orthogonal to the FP6 input
formats and works unchanged.

Producing FP6 data (quantize + pack + store) is host-side today: tests pack
on the CPU and pass uint8 tensors to the kernel. A device kernel that wanted
to emit FP6 would `tl.store` (or descriptor-store) plain uint8 bytes; the
bytes-in/bytes-out nature of the store paths means no compiler support is
required beyond what uint8 already has.

SWP (software pipelining: the temporal prologue / steady-state / epilogue
pipelining of loop bodies, including pipelined stores such as AMD's TDM
store pipeline) is likewise unaffected: store pipelining operates on the
compute-type outputs, and FP6 inputs only appear on the load side of the
loop.

## 5. Layouts without NPOT, and with NPOT

### Without NPOT support (today)

Tensor layouts are built on power-of-two bases, so block shapes are powers
of two unless NPOT is explicitly enabled (gated by `TRITON_ALLOW_NPOT=1` /
`knobs.language.allow_npot`, enforced in the frontend shape checks). FP6 fits
this world as follows:

- The tensor-level divisibility rule is K % 16 == 0 (one 96-bit dense
  group), which every power-of-two K >= 16 already satisfies; in practice
  blocks use K = 64/128. Host packing by itself only requires % 4.
- Kernels load the dense global stream into logical-width tiles, so
  blocked/linear layouts over the uint8 tensor need no new basis kinds.
- The shared-memory MMA operand view is the one place that reasons in packed
  bytes rather than logical elements: instruction K extents and K coordinates
  are converted with x3/4 math (sections 7 and 9). Byte-K extents such as 48
  or 96 are already non-powers-of-two, and the descriptor builder compares
  exact extents (not log2 sizes) precisely so these packed extents work.
- The WMMA FP6 instruction shape `[16, 16, 96]` shows the same principle at
  the instruction level: logical K 96 (a multiple of 16, not a power of two)
  with packed byte-K 72.

### If/when NPOT support is added

The FP6 packing math itself is NPOT-agnostic: the host-side % 4 packing
and the lowering's intra-tile % 4 step math work for any multiple of 4,
but the scaled-dot tensor contract requires multiples of 16, so K values
like 48 or 96 (multiples of 16, not powers of two) would flow through
packing, scale-shape deduction, and the smem/TMEM coord mapping
unchanged once NPOT shapes are allowed. What NPOT enablement would
additionally require is all in generic infrastructure, not FP6-specific:

- Block tiling and swizzled/padded shared layouts that assume power-of-two
  bases would need NPOT-aware bases for the affected dims.
- TMA/TDM descriptors keep their hardware granularity (16-byte alignment and
  box/stride multiples; 256-byte TDM inner-dim guidance), so some NPOT tile
  shapes would still be undescribable and must fall back to non-descriptor
  copies or padding.
- Autotuning search spaces and the NPOT error-pruning hooks would need FP6
  configs included once the infrastructure allows them.

## 6. CUDA-core operations: quant and dequant

Dequantization (FP6 codes to compute type) is pure elementwise integer/float
arithmetic, i.e. CUDA-core / VALU work, in three places:

- **Compiler decompose path** (`upcastFp6` / `fp6BitsToFloat`): gathers the
  3-byte groups, extracts 6-bit codes with shifts and masks, and converts
  each code exactly: normals via `(1 + m/2^mbits) * 2^(e-bias)` times the
  sign, subnormals via `(m/2^mbits) * 2^(1-bias)`. Every intermediate value
  is exactly representable, so computing directly in f16/bf16 is exact.
  Scales are applied afterwards through the existing scale path.
- **Interpreter**: decodes E2M3 (2 exponent bits, bias 1) and E3M2
  (3 exponent bits, bias 3) to f32 with the same normal/subnormal split,
  asserting K-packed inputs.
- **Reference/test code**: exact OCP MX FP6 decode to f32 on the host.

There is deliberately no device-side quantize (float to FP6 pack) op: FP6
production is host-side only. If one is ever needed, it would be the mirror
image -- elementwise scale, clamp, round, and bit-pack into the LSB-first
3-byte groups -- followed by a plain uint8 store.

## 7. Shared memory (NVIDIA) and LDS (AMD)

### What runs today

The decompose path upcasts FP6 to f16/bf16 before the dot, so shared
memory / LDS on the executed path holds plain 16-bit operands and the
existing MMA operand handling applies unchanged. FP6 bytes exist in flight
only as uint8 in global memory, in transit through pipelined buffers, or
transiently in registers.

### Native-path lowering design (ready, gated off)

The lowering already contains the K math a native FP6 MMA path needs, kept
in packed bytes end to end:

- **NVIDIA smem** (`DotOpMmaSmemLoader` with `isFp6`): the memdesc layout
  stays in packed bytes; the instruction K extent is converted to bytes
  (`instrK * 3 / 4`, asserting `instrK % 4 == 0`), and logical K coordinates
  are converted to byte coordinates in `smemLoad`, with `fp6Trans` handling
  the case where the packed dim is the transposed one. The smem base keeps
  its 16-byte alignment (offsets computed in 128-bit units).
- **AMD MFMA** (`packedKBase`): `kBase * 3 / 4` with a `kBase % 4 == 0`
  assert.
- **AMD WMMA** (`packedKDim`): `kInstrSize * 3 / 4` with the same assert.
- **AMD WmmaGroup**: `kDim * 3 / 4` per operand for the same packed-K
  accounting, plus the `wmmaPacked6Enc` encoding for FP6 operands.

Reads and writes of the packed operand through smem/LDS are therefore
ordinary uint8 accesses under a layout expressed in packed bytes; the x3/4
mapping lives at the MMA operand boundary, not in the memory subsystem.

## 8. Scaled dot operations

End to end, `tl.dot_scaled` with an FP6 operand compiles as follows:

1. **Frontend** validates formats, requires K-packed uint8 storage for FP6,
   computes unpacked dims (identity for FP6: tensor shapes count logical
   values, asserted % 16 for the 96-bit dense group), and deduces scale
   factors with FP6-aware K divisibility.
2. **IR** `tt.dot_scaled` carries `ScaleDotElemType::E2M3`/`E3M2`; verifiers
   reject K dimensions not divisible by 16 (with K-pack) and non-i8 storage
   on NVIDIA.
3. **Accelerator matching refuses native FP6 MMA**: the NVIDIA MMAv5 pattern
   fails with "fp6 uses decompose upcast" ("fp6 cannot tile the tcgen05 K
   dimension today"), and the AMD MFMA/WMMA patterns likewise route FP6 to
   the decompose path ("NYI: fp6 without K-packing" for the unsupported
   un-packed case).
4. **Decompose** (`DecomposeScaledBlocked`, with an AMD subclass delegating
   FP6 to the base gather-based upcast since there is no fused scaled-upcast
   op for FP6) upcasts each FP6 operand to the compute type, applies scales,
   and rewrites the op as a plain dot.
5. **Codegen** emits standard `tc_gen5_mma` (sm100), `v_mfma_f32_*` (gfx950),
   or `v_wmma_f32_*` (gfx1250) over f16/bf16 operands; FP6 upcast helpers are
   inline arith. ASM checks assert the absence of `mxf8f6f4` / `f8f6f4`.

The native-path K math in sections 7 and 9 exists so that a future native
FP6 instruction (or direct Gluon construction of an FP6 MMA op) can reuse
the same packing contract; the Triton `tl.dot_scaled` path does not exercise
it today.

## 9. TMEM reads and writes

On Blackwell, TMEM holds MMA operands and accumulators for `tcgen05.mma`.
The MMAv5 TMEM loader is parameterized by storage vs logical bit width and
handles FP6 explicitly:

- `isFp6` is defined as 8-bit storage with 6-bit logical elements, and the
  loader's validity assert is relaxed accordingly: normally the logical bit
  width must divide the storage bit width, but 6 does not divide 8, so FP6
  takes its own branch.
- `tmemLoad` receives a logical K coordinate and converts it to the packed
  byte coordinate (`b * 3 / 4`, asserting `b % 4 == 0`), mirroring the smem
  loader: TMEM operand data is addressed in packed bytes under a layout
  expressed in packed bytes, and the memdesc layout absorbs any additional
  physical padding.
- Accumulator TMEM traffic is unaffected: accumulators are f32 regardless of
  input format, and the decompose path that runs today never places FP6 in
  TMEM at all (operands arrive already upcast).

As with smem, this is native-path design: it specifies exactly how FP6 would
be addressed in TMEM (packed bytes, x3/4 logical-to-physical K mapping, % 4
alignment invariants) for a future native instruction, while current
execution keeps TMEM FP6-free via decompose-then-upcast.

## Status and limitations

- FP6 requires K-packing; non-K-packed FP6 is rejected ("NYI: fp6 without
  K-packing").
- K must be a multiple of 16 at the tensor level everywhere (16 values
  per 96-bit / 12-byte dense group); 2D TMA descriptors need the same
  step. % 4 steps remain only as intra-tile lowering math.
- No packed FP6 stores and no device-side FP6 quantize op exist; FP6
  production is host-side.
- Native FP6 MMA (tcgen05/MFMA/WMMA consuming 6-bit data) is designed in the
  lowering but gated off; execution goes through exact upcast to f16/bf16.
