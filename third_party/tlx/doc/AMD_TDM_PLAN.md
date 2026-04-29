# AMD TDM Support Plan for TLX

Plan for landing AMD gfx1250 Tensor Data Movement (TDM) support in TLX as a
sequence of stacked commits on a single working branch (`tlx-tdm`). Each
stage below corresponds to one commit; the stack is rebased / squashed before
upstreaming so each stage stays atomically reviewable.

## Status

- **Prerequisite landed:** PR3 of the prior landing plan
  (`InsertRequireLayout` rewritten on top of `SparseBackwardDataFlowAnalysis`,
  commit `754023ea0`). The new layout-propagation infrastructure is the right
  place to plug TDM-friendly shared encodings.
- **Out of scope:** TLX's NV TMA bindings are currently inert
  (`create_async_TMA_load` / `_store` / `_reduce` in `triton_tlx.cc` are
  commented out as "TODO: upstream signature changed"). This plan does not
  touch them — the AMD path takes a separate C++ binding so the NV stub stays
  broken in exactly the same way as today, to be revived independently.

## Goal

Let TLX kernels target gfx1250's TDM hardware (the
`amdgpu.async_tdm_copy_global_to_local` /
`amdgpu.async_tdm_gather` / `amdgpu.async_tdm_scatter` /
`amdgpu.async_tdm_prefetch` family lowered to
`llvm.amdgcn.tensor.load.to.lds` and `store.from.lds`) by reusing the base
Triton AMD lowering pipeline. Add TLX surface only where there is no
portable analogue.

## Background

### How base Triton reaches TDM
- Frontend descriptor load (`tl.load(desc, ...)` or `tt.DescriptorLoadOp`)
  survives `make_ttir` because `compiler.py` skips
  `add_rewrite_tensor_descriptor_to_pointer` on TDM-capable arches.
- `add_convert_to_tensor_ops` rewrites `tt.DescriptorLoadOp` →
  `amdgpu.async_tdm_copy_global_to_local` + `amdgpu.async_tdm_wait` +
  `ttg.local_load`.
- `add_pipeline` software-pipelines TDM chains; `streamPredication` converts
  i1 predicates to i32 for TDM ops.
- `add_update_async_wait_count` rewrites `amdgpu.async_tdm_wait` →
  `amdgpu.async_tdm_intrinsic_wait` with summed intrinsic counts.
- LLVM lowering (`LoadStoreOpToLLVM.cpp`, `TDMUtility.cpp`) emits the
  `tensor.load.to.lds` intrinsics, splitting per partitioned-shared layout.

### Where TLX stands today
- TLX dialect ops are layout/aliasing helpers only — no async/TDM ops.
- All memory-movement Python builtins emit existing TTG/NV-GPU ops:
  `tlx.async_load` → `ttg.AsyncCopyGlobalToLocalOp`,
  `tlx.local_load` → `ttg.LocalLoadOp`, etc.
- `tlx.async_descriptor_load` → `ttng.AsyncTMACopyGlobalToLocalOp` (NV-only),
  but the C++ binding is currently a no-op stub.
- One AMD-specific touch point: `tlx.local_load(..., relaxed=True)` stamps
  `ttg.amdg.syncedViaAsyncWait` on the `ttg.LocalLoadOp` so the AMD backend
  can elide a `vmcnt(0)` wait in 3-stage pipelines.
- The async path through `tlx.async_load` already reaches
  `amdgpu.buffer_load_to_local` on CDNA3/4 via
  `add_convert_to_buffer_ops`. **gfx1250 TDM is not reachable from any TLX
  builtin.**

## Gap

| Capability | Base Triton AMD | TLX |
|---|---|---|
| Descriptor load → TDM copy | `tt.DescriptorLoadOp` → `ConvertToTensorOps` | absent (NV TMA stub) |
| TDM gather/scatter | `amdgpu.async_tdm_gather/scatter` | absent |
| TDM prefetch | `amdgpu.async_tdm_prefetch` | absent |
| TDM local→global store | `amdgpu.async_tdm_copy_local_to_global` | absent |
| TDM wait (outer) | `amdgpu.async_tdm_wait` | absent (only generic `ttg.async_wait`) |
| TDM wait (intrinsic count) | `amdgpu.async_tdm_intrinsic_wait` | absent |
| TDM-friendly shared encodings | `PaddedSharedEncodingAttr`, `PartitionedSharedEncodingAttr` | not exposed in `tlx.local_alloc` |

## Commit stack

Stages land in order on `tlx-tdm`, one commit each. Each commit is
independently reviewable and leaves the branch buildable + tested.

### Stage A — AMD TDM load + wait

**Files:**
- `third_party/tlx/dialect/triton_tlx.cc`
- `third_party/tlx/language/tlx/mem_ops.py`
- `third_party/tlx/language/tlx/utility.py`
- `third_party/tlx/language/tlx/__init__.py`
- `python/test/unit/language/test_tlx_amd.py`

**Changes:**
- Add a new C++ binding `create_async_tdm_copy_global_to_local` that
  emits `amdgpu::AsyncTDMCopyGlobalToLocalOp` directly into the user's
  buffer.
- Add a new C++ binding `create_async_tdm_wait` emitting
  `amdgpu::AsyncTDMWait`.
- Add a `tlx.async_tdm_load(desc, result, offsets, pred=...)` Python
  builtin (AMD TDM target only — asserts `is_amd_tdm_target(arch)`).
  The signature is intentionally narrower than `async_descriptor_load`:
  no `barrier` (counter-based), no `cache_modifier` /
  `eviction_policy` / `multicast_targets` (NV-only). Returns an
  optional `tlx.async_token` for use with `async_tdm_wait`.
- Add a `tlx.async_tdm_wait(pendings, tokens=[])` Python builtin
  emitting `amdgpu.async_tdm_wait`. Prefer this over overloading
  `tlx.async_load_wait_group` to keep the wait domain explicit at the
  source level.
- Do not touch `tlx.async_descriptor_load` — its NV TMA path is a
  separate upstream concern. AMD users get a distinct, AMD-only API
  surface (`async_tdm_*`) with no target-conditional behavior.
- Compile tests on gfx1250: assert `amdgpu.async_tdm_copy_global_to_local`
  and `amdgpu.async_tdm_wait` in TTGIR; assert `tensor.load.to.lds` in
  AMDGCN. Cover both the count-only and token-threaded wait paths.

**Direct emission rationale.** TLX's load operations all take a
user-provided `result: buffered_tensor`, so the natural target is the
async op that takes a buffer destination — `amdgpu::AsyncTDMCopyGlobalToLocalOp`
— not `tt.DescriptorLoadOp`. Going through `tt.DescriptorLoadOp` would
force `ConvertToTensorOps` to allocate a fresh buffer, insert a
synchronous `AsyncTDMWait num=0`, and round-trip through registers via
`LocalLoadOp`, defeating the user's explicit buffering and async
intent.

**Separate API rationale.** Reusing `async_descriptor_load` would force
target-conditional argument shapes and return types into a single
builtin (NV requires `barrier`, AMD forbids it; NV returns `None`, AMD
returns a token; NV honors `cache_modifier`/`eviction_policy`/
`multicast_targets`, AMD does not). That's a leaky abstraction maintained
by runtime asserts. A distinct `async_tdm_*` family keeps each call site
honest about which hardware it targets, and matches the precedent set by
Gluon's `ttgl.amd.gfx1250.tdm.*` namespace.

**Why first:** Smallest stage with end-to-end value. After this commit
the AMD-only TDM load case compiles on gfx1250 and exercises the
downstream AMD pipeline (`Pipeline` → `UpdateAsyncWaitCount` → LLVM
lowering). `ConvertToTensorOps` is bypassed for this path, but it still
runs to catch any stray `tt.DescriptorLoadOp` (e.g., from
`tl.load(desc, ...)`).

**Known Stage A limitation (driven Stage B).** The default
`tlx.local_alloc` returned a `SwizzledSharedEncodingAttr(maxPhase=1)`
buffer. The TDM op verifier accepted this (its only check is "no
swizzling"), so the kernel compiled cleanly, but on real gfx1250
hardware `tensor_load_to_lds` requires a `PaddedSharedEncodingAttr`
(or `PartitionedSharedEncodingAttr` containing one) to lay out LDS
correctly — without it the load runs, the wait completes, and the
subsequent `local_load` reads back garbage. Resolved in Stage B by
exposing `tlx.padded_shared_layout_encoding`; users now pass an
explicit `layout=` to `tlx.local_alloc`.

### Stage B — TDM-aware shared layouts in `local_alloc` ✓

**Files:**
- `third_party/tlx/language/tlx/types.py`
- `third_party/tlx/language/tlx/mem_ops.py`
- `third_party/tlx/language/tlx/__init__.py`
- `third_party/tlx/dialect/triton_tlx.cc`
- `python/test/unit/language/test_tlx_amd.py`

**Changes (landed):**
- Expose `tlx.padded_shared_layout_encoding`, the identity-mapping form
  of `ttg.padded_shared_encoding`. Mirrors Gluon's
  `PaddedSharedLayout.with_identity_for(intervals, shape, order)`.
- Bind `make_padded_shared_encoding_attr` in `triton_tlx.cc`, calling
  the second `PaddedSharedEncodingAttr::get(...)` builder
  (`intervalPads`, `order`, `shape`, `cgaLayout`).
- Wire `tlx.local_alloc(..., layout=...)` to honor a user-supplied
  `tlx.shared_layout_encoding` (replaces the previous
  `NotImplementedError`). Default-layout behavior is unchanged.

**Deferred (follow-up).** The plan originally proposed auto-defaulting
`tlx.local_alloc` to a TDM-friendly encoding when the buffer is
consumed by a TDM op. That requires forward-analysis at allocation
time; Gluon also doesn't do it (every `tdm.async_load` test passes an
explicit layout). Stays as user-explicit for now. Layout-propagation
support for `PaddedSharedEncodingAttr` through `MemDescTransOp` is
also deferred until a kernel actually needs the transpose path.

### Stage C — Descriptor plumbing edges

**Files:**
- `third_party/tlx/language/tlx/mem_ops.py`
- `third_party/tlx/language/tlx/__init__.py`
- `third_party/tlx/dialect/lib/Transforms/Fixup.cpp` (audit, deferred)
- `python/test/unit/language/test_tlx_amd.py`

**Changes (landed):**
- Add `tlx.descriptor_compatible_layout(desc)` builtin: returns a
  shared-memory layout matching the AMD TDM descriptor's expected
  encoding (the same formula upstream's
  `AMDGPUAssignDescriptorMemoryLayouts::buildFallbackSharedEncoding`
  applies). Replaces hand-coded `with_identity_for([(N, M)], ...)`
  call sites.
- Factor the descriptor-layout formula out of the runtime warning into
  a private `_amd_tdm_descriptor_layout(desc)` helper shared by
  `descriptor_compatible_layout` and the `async_tdm_load` mismatch
  guard. Comment in the helper points to the upstream source of
  truth so the formula stays in sync.

**Backward layout propagation (landed alongside this stage):**
- Extended `TLXInsertRequireLayout` to walk
  `amdgpu.async_tdm_copy_global_to_local` ops and anchor a
  `tlx.require_layout` on the buffer operand using the
  descriptor-compatible padded encoding. The existing
  `tlx-propagate-layout` pass picks up the constraint via its already-in-place
  `LayoutBackwardPropagation` lattice and rewrites the source
  `local_alloc` (plus subview / loop-carrier chain) to match.
- Exposed `triton::amdgpu::buildDefaultTDMDescriptorEncoding` as a public
  helper in `third_party/amd/lib/TritonAMDGPUTransforms/Utility.h`; refactored
  `AMDGPUAssignDescriptorMemoryLayouts::buildFallbackSharedEncoding` to
  forward to it, so the formula is single-sourced. TLX's transforms lib
  links `TritonAMDGPUTransforms` and calls the helper directly.
- Effect: `tlx.local_alloc(..., /*no layout=*/)` followed by
  `tlx.async_tdm_load(desc, smem, ...)` is now correct by construction
  on gfx1250 — the alloc's encoding gets rewritten to
  `padded_shared<[N:+M], ...>` automatically. The Python runtime warning
  remains as a backstop for users who pass an explicit
  *non-default-and-non-matching* layout.

**Deferred (follow-up):**
- Audit `TritonTLXFixup` to verify it does not strip metadata from
  `tt.DescriptorLoadOp` / `tt.MakeTensorDescOp`.
- Make `cache_modifier` and `eviction_policy` no-ops on AMD with a
  diagnostic (TDM intrinsics do not accept them).
- Drop `multicast_targets` on AMD with a diagnostic (NV-only concept;
  AMD has `cluster_load_async_to_lds` but it's not exposed via the
  descriptor surface).
- Extend the TDM-anchor walk to the deferred siblings
  (`async_tdm_copy_local_to_global`, gather, scatter, prefetch) once
  those builtins land.

### Stage D — Tutorials + docs

**Files:**
- `third_party/tlx/tutorials/amd-gemm-pipelined-gfx1250.py` (new)
- `third_party/tlx/doc/AMD_TDM_PLAN.md` (this doc, mark stages done)

**Changes:**
- Pipelined GEMM on gfx1250 using `tlx.async_descriptor_load` +
  `tlx.async_descriptor_wait`.
- Document the descriptor API, wait semantics, and layout selection
  choices in this doc.

## Deferred (follow-up work)

Out of scope for the initial landing, but enumerated here so we don't
forget them.

### Explicit AMD TDM builtins

The AMD `amdgpu` dialect has TDM ops that are not reachable through
any portable Triton-IR descriptor op. Once stages A–D are stable we
can add direct TLX bindings:

- `tlx.async_tdm_gather(desc, result, offsets, indices, pred=None)`
  → `amdgpu.AsyncTDMGatherOp`
- `tlx.async_tdm_scatter(desc, src, offsets, indices, pred=None)`
  → `amdgpu.AsyncTDMScatterOp`
- `tlx.async_tdm_copy_local_to_global(desc, src, offsets, pred=None)`
  → `amdgpu.AsyncTDMCopyLocalToGlobalOp` (TDM store direction)
- `tlx.async_tdm_prefetch(desc, offsets, pred=None)`
  → `amdgpu.TDMPrefetchOp`
- `tlx.async_tdm_intrinsic_wait(count)`
  → `amdgpu.AsyncTDMIntrinsicWait` (fine-grained escape hatch)

These are AMD-only with arch asserts. They unlock attention kernels
with gather-to-LDS and TDM-pipelined producers that need explicit
prefetch.

### Pipeline integration audit (only if explicit builtins land)

Once explicit TDM builtins exist, audit the AMD passes to confirm
they accept TLX-originated TDM ops not produced by `ConvertToTensorOps`:

- `LowerLoops::initSchedule` should recognize the resulting TDM chains.
- `countTDMInstructions` should cover them.
- `streamPredication` should cover any predicate widening they need.

Until the explicit builtins land this audit is unnecessary —
the only TDM ops in flight come from `ConvertToTensorOps`, which the
AMD passes already handle by construction.

## Stage order

```
A  async_tdm_load + async_tdm_wait
B  TDM layouts in local_alloc
C  descriptor plumbing edges
D  tutorials + docs
```

Stages are committed in this order on `tlx-tdm`. B and C only depend
on A; if work proceeds in parallel they may swap order in the final
stack as long as the post-rebase sequence keeps each commit
buildable. D is always last.

Deferred items (gather/scatter/prefetch/store builtins, pass
integration audit) follow once the descriptor-load path is proven.

## Key design decisions

1. **Reuse base Triton's TDM ops, do not fork them.** Stage A and the
   deferred explicit builtins all emit the existing
   `amdgpu.async_tdm_*` op set directly. No new MLIR dialect ops in
   TLX. The only piece of `ConvertToTensorOps` we skip is the implicit
   alloc+wait+local_load wrapper it would generate around
   `tt.DescriptorLoadOp` — TLX users have already made the buffer and
   wait choices explicitly.
2. **Do not entangle with the broken NV TMA path.** Stage A adds new
   `async_tdm_*` Python builtins and `create_async_tdm_*` C++ bindings
   rather than reviving `create_async_TMA_load` or grafting AMD logic
   onto `async_descriptor_load`. The NV stub stays inert; whoever
   re-enables NV TMA does that work independently against the
   `async_descriptor_load` API. AMD and NV have separate, narrowly-typed
   API surfaces.
3. **Wait domain explicit at source level.** Add `tlx.async_tdm_wait`
   alongside `tlx.async_load_wait_group` rather than overloading the
   latter. Mixing TDM and async-copy ops in the same loop is rare; on
   the rare occasion it matters the user picks the matching wait.
4. **AMD-only builtins use plain `tlx.*` namespace with the
   `async_tdm_` prefix.** Matches `_assert_blackwell_for_tmem` (flat
   namespace + arch assertions); the `async_tdm_` prefix encodes the
   hardware/wait domain in the name itself, making each call site
   self-documenting and unambiguously AMD-only without a `tlx.amd.*`
   namespace.
5. **No target-conditional builtins.** `async_tdm_load` does not also
   serve NV; `async_descriptor_load` does not also serve AMD. The
   alternative — one name with arch-conditional argument shapes and
   return types — is a leaky abstraction that compounds across deferred
   stages (gather/scatter/store/prefetch are AMD-TDM-only).

## Open questions

1. Should `tlx.local_alloc` on AMD gfx1250 default to a TDM-friendly
   encoding speculatively, or only when the compiler can prove the
   buffer is consumed by a TDM op? Stage B currently chooses the
   latter (analyze, then default).
2. `cluster_load_async_to_lds` (gfx1250 multi-CTA) — TLX surface or
   strictly internal? Out of scope for this plan; revisit after the
   deferred explicit-builtin work.

## Verification

Each stage commits:
- Lit tests under `test/TLX/` and/or `third_party/amd/test/` for the
  MLIR-level lowering it enables.
- pytest compile tests in `python/test/unit/language/test_tlx_amd.py`
  using `triton.compile(ASTSource(...), target=GPUTarget("hip",
  "gfx1250", 32))` and asserting the expected MLIR ops in TTGIR and
  the expected AMDGCN intrinsics:
  - `tensor.load.to.lds` for TDM copies
  - `s_wait_tensorcnt` (or platform equivalent) for waits
- Correctness tests behind a hardware availability gate, mirroring the
  existing `is_gfx950_available()` pattern.

Compilation tests run on any host with HIP. Correctness tests run on
gfx1250 hardware only.
