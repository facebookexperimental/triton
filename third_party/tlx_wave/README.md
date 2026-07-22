# TLX Wave backend

`tlx_wave` is a Triton backend that compiles TLX kernels for AMD GPUs through
the [Wave](../wave/README.md) compiler. It reuses Triton's TLX/TTIR frontend and
AMD TTGIR pipeline, translates the resulting program into structural
Wave/WaveAMD IR, and lets Wave perform target scheduling, register allocation,
and machine lowering.

The backend currently targets `gfx942` and `gfx950` in wave64 mode. It is a
separate backend, selected with `TRITON_DEFAULT_BACKEND=tlx_wave`; it does not
silently fall back to the AMD LLVM lowering when an operation is unsupported.

## Build and run

Initialize the Wave submodule and build Triton from the repository root:

```bash
git submodule update --init --recursive third_party/wave
MAX_JOBS=16 make dev-install
```

The first build may also build Wave's pinned LLVM/MLIR with Python bindings and
a standalone Wave tree. `MAX_JOBS` controls the default parallelism used by
those builds. A direct CMake configuration must include `tlx_wave` in
`TRITON_CODEGEN_BACKENDS`.

Select the backend when running a TLX program:

```bash
TRITON_DEFAULT_BACKEND=tlx_wave python path/to/program.py
```

The compiler exposes the high-level Wave module as `compiled.asm["wave"]` and
the executable ELF object as `compiled.asm["hsaco"]`. Bridge statistics and
policy settings are also recorded in compilation metadata under the
`tlx_wave_*` keys.

## Compilation pipeline

```text
TLX / Triton Python
        |
        v
      TTIR                 TLX fixup, alias lowering, canonicalization
        |
        v
 AMD TTGIR                 standard HIP layout and optimization pipeline
        |
        v
 prepared TTGIR            warp pipeline conversion and AMD membar insertion
        |
        v
 structural bridge         import -> types/layouts -> facts -> tokens
        |                            -> target plan -> verify -> emit
        v
 Wave / WaveAMD IR         symbolic gather, scatter, redistribute, MMA, DMA
        |
        v
 Wave target pipeline      scheduling, register allocation, machine lowering
        |
        v
      HSACO                 loaded and launched through the HIP driver
```

Triton owns the source language semantics, TTIR/TTGIR formation, AMD warp
pipeline conversion, and compiler membar analysis. The bridge preserves that
contract in Wave's high-level dialects. Wave owns the implementation of
high-level Wave operations and all target-machine decisions after the handoff.

The bridge is implemented in Python because Triton's compiler and the Wave
submodule use separate LLVM/MLIR builds. MLIR objects from one build are never
passed into the other. Instead, the bridge snapshots TTGIR into an immutable,
implementation-independent schema and constructs Wave IR with Wave's structural
Python bindings.

## Bridge design

The bridge is a staged converter, not an emitter that discovers semantics while
printing operations. Its entry point is
[`converter/pipeline.py`](backend/converter/pipeline.py), and each stage has a
closed responsibility:

| Stage | Main modules | Responsibility |
| --- | --- | --- |
| Source import | [`source_import.py`](backend/converter/source_import.py), [`source_ir.py`](backend/converter/source_ir.py) | Copy kernel, region, operation, value, type, and attribute data out of TTGIR. Source MLIR handles do not cross this boundary. |
| Type and layout conversion | [`types.py`](backend/converter/types.py), [`layouts.py`](backend/converter/layouts.py), [`coordinates.py`](backend/converter/coordinates.py), [`layout_remap.py`](backend/converter/layout_remap.py) | Choose Wave value representations and derive symbolic named-dimension layout relations. |
| Fact analysis | [`facts.py`](backend/converter/facts.py) | Record proven ranges, divisibility, affine coordinates, and pointer byte ranges with provenance. It does not assume integer arithmetic is non-overflowing without proof. |
| Token analysis | [`tokens.py`](backend/converter/tokens.py) | Build async groups, memory effects, users, and structured-control-flow token carries. This is the sole owner of source dependency analysis. |
| Operation conversion | [`op_conversion.py`](backend/converter/op_conversion.py), [`domains.py`](backend/converter/domains.py) | Rewrite source operations into a schema-closed `TargetProgram` with explicit operands, results, attributes, regions, facts, layouts, and event domains. |
| Target cleanup and ordering | [`canonicalize.py`](backend/converter/canonicalize.py), [`barrier_order.py`](backend/converter/barrier_order.py) | Apply target-level canonicalization and encode sparse issue ordering around full barriers. |
| Verification | [`verifier.py`](backend/converter/verifier.py) | Reject incomplete plans, invalid representations, escaped fragments, malformed token domains, or unsupported semantics before emission. |
| Structural emission | [`emission.py`](backend/converter/emission.py) | Mechanically create Wave/WaveAMD operations with Wave's builders. It does not inspect the source graph or choose a lowering family. |

`ConversionOutput` retains the imported source, type/layout, fact, token, target,
and emitted layers. This makes failures testable at the boundary where they are
introduced rather than only after machine-code generation.

### Boundary invariants

These rules define the bridge contract:

- Source operations and values are identified by stable integer IDs after
  import. No Triton MLIR object, callback, layout object, or emitter helper may
  leak into target operations or attributes.
- Target attributes contain only closed data such as primitives, tuples, and
  frozen sets. The target verifier must be able to validate the complete plan
  without consulting the source MLIR module.
- Operation rewriters are stateless. Any semantic relationship they need is
  supplied by the type/layout, fact, or token stages.
- Emission is structural. Handwritten or parsed Wave text is not a supported
  lowering path, and emission must not rediscover producer/consumer semantics.
- Unsupported source semantics produce a structured `TLXW_*` diagnostic with
  stage and source/target provenance. There is no approximate lowering and no
  LLVM fallback.
- Existing TLX kernel semantics are preserved. Source kernels must not need
  extra `tl.debug_barrier()` calls to become correct on this backend.

### Values, layouts, and movement

Distributed layouts are modeled as named-dimension maps from physical
dimensions such as register, lane, warp, and block to logical tensor
coordinates. Shared layouts similarly map an LDS offset and block coordinate to
logical coordinates. The model covers blocked and linear layouts, slices, dot
operands, AMD MFMA layouts, and supported shared swizzle, padding, partition,
and rotation forms. A relation that cannot be proved from imported layout data
is rejected rather than guessed.

Uniform scalars and pointers remain scalar. Lane-varying values become
`wave.simd` values, with tuples used when a source tensor has multiple register
components. Masks and per-lane pointers use corresponding structural
representations.

Layout movement stays high level across the bridge:

- `ttg.convert_layout` becomes an alias when the representations are identical,
  or one symbolic `wave.redistribute` relation otherwise. Wave decides whether
  the relation is implemented with shuffles, gathers, LDS, or another target
  mechanism.
- Local reads and writes become symbolic `wave.gather` and `wave.scatter`
  operations where applicable. Masking and zero fill are represented
  structurally around those operations; the bridge does not pre-partition
  accesses into target-specific cases.
- MFMA fragments are an immediate WaveAMD instruction-lowering detail. Ordinary
  typed SIMD packets cross operation boundaries; fragment values may exist only
  while packing operands for or unpacking results from `waveamd.mma`.

See [TLXWaveBridgeLayoutSystem.md](backend/TLXWaveBridgeLayoutSystem.md) for the
layout algebra, supported encoding families, quotienting rules, and rejection
policy.

### Async memory and barriers

The token plan preserves the source async protocol explicitly:

- An async DMA is completed only by an explicit source `ttg.async_wait`
  (`async_load_wait_group`). Neither an LDS alias, a destination buffer, an
  allocation history, an ordinary barrier, nor an implicit dependency may
  complete it.
- `wait_group(K)` completes the older committed groups and leaves the newest
  `K` groups live. Retained groups contribute issue ordering only; DMA packets
  within a group remain siblings and are not accidentally serialized.
- Readiness produced by a wait is threaded through local-memory consumers and
  through `scf.for`/`scf.if` arguments, results, and yields. An absent branch
  event is represented by a neutral token.
- Compiler-inserted membars are preserved unless target canonicalization proves
  that a specific compiler barrier is redundant with an adjacent wait
  publication barrier. User/source barriers are not removed by that rule.
- A full-memory barrier uses sparse issue ordering: pre-barrier memory issuers
  feed the barrier through issue tokens, and later memory issuers consume the
  post-barrier issue epoch. This orders issue without pretending to complete
  DMA. Pure arithmetic, MMA, reductions, and layout redistribution are not
  pinned by the full-barrier epoch.

Although these event classes eventually use `!wave.mem.token`, the target plan
keeps distinct domains for DMA completion, DMA groups and issue, ordinary
memory completion and issue, barrier issue, LDS readiness/frontiers, and neutral
events. Verification prevents one domain from being used as another.

The complete protocol and its control-flow examples are in
[TLXWaveAsyncMemoryProtocol.md](backend/TLXWaveAsyncMemoryProtocol.md).

## Configuration

The following compile options are opt-in and are attached to the individual
kernel function before bridge import:

| Compile option | Environment default | Effect |
| --- | --- | --- |
| `tlx_wave_enable_split_barriers=True` | `TRITON_TLX_WAVE_ENABLE_SPLIT_BARRIERS=1` | Enables WaveAMDMachine split-barrier lowering for eligible barriers. |
| `tlx_wave_enable_multi_wave_specialize=True` | `TRITON_TLX_WAVE_ENABLE_MULTI_WAVE_SPECIALIZE=1` | Enables Wave's joint multi-wave specialization for that kernel. |

An explicit compile option overrides its environment default. Both settings
default to disabled.

Tool discovery normally uses the standalone build under `third_party/wave`.
Development builds can override it with:

- `TRITON_WAVE_OPT`: path to `wave-opt`.
- `TRITON_WAVE_TOOLS_DIR`: directory containing Wave tools.
- `TRITON_WAVE_PYTHONPATH`: Wave Python binding/package search path.
- `TRITON_WAVE_PIPELINES`: path to Wave's `pipelines.mlir` transform library.

## Testing and performance

After compiler or native changes, rebuild before running tests:

```bash
make
python -m pytest -n auto -s --tb=short \
  python/test/unit/language/test_tlx_wave_backend.py
python -m pytest -n auto -s --tb=short \
  third_party/tlx/tutorials/amd-gemm-warp-pipeline-tlx-wave_test.py
```

The unit suite exercises individual bridge stages, structural invariants,
diagnostics, generated Wave IR, HSACO loading, and representative GPU kernels.
The unified LLVM-versus-Wave performance sweep covers f16 and MXFP GEMMs, GLU,
and flash attention:

```bash
python third_party/tlx/tutorials/run_wave_perf_sweeps.py
```

Use the sweep's `--help` output for kernel subsets, input distributions,
parallel compilation, timing methodology, and Wave policy flags. Performance
changes to the bridge should be checked against the full sweep, because a
target-specific shortcut that helps one layout can regress unrelated kernels.

## Contributor map

- [`backend/compiler.py`](backend/compiler.py): Triton backend stages, options,
  pre-bridge passes, metadata, and Wave/HSACO handoff.
- [`backend/driver.py`](backend/driver.py): target selection and HIP executable
  loading.
- [`backend/wave_bridge_tools.py`](backend/wave_bridge_tools.py): Wave binding,
  tool, verification, and compilation discovery.
- [`backend/converter/`](backend/converter/): the structural TTGIR-to-Wave
  bridge.
- [`backend/AGENTS.md`](backend/AGENTS.md): mandatory local development and
  correctness rules.
- [`../wave/`](../wave/): the pinned upstream Wave compiler submodule.
