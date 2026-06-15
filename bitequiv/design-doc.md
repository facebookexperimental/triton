# bitequiv — Project Design Doc

> **This is the design doc for the whole `bitequiv` project** (bitwise equivalence &
> constraint-aware autotuning), written for humans. It is the single reference for:
> the problem, the architecture, the core technical model, the equivalence
> algorithm, the **public interface (how to use it and how to enable it)**, and the
> **rules future development follows**.
>
> **When asked to extend `bitequiv`, follow this doc — and keep it in sync** when the
> design, algorithm, or interface changes. Terse machine-readable state lives in
> `bitequiv/PROGRESS.md`; the project guide/conventions live in `bitequiv/CLAUDE.md`.

Status: M1 (2026-06-15). MLIR-native TTGIR checker over distributed encodings
(`toLinearLayout`), single- and multi-operand `tt.reduce`, plus the **PTX/FMA
backstop** (§6.5). Precise MMA/`tl.dot` accumulation order is future work (§7).

---

## 1. What this project is, and why

Two intertwined goals:

1. **Bitwise equivalence.** Two kernels are bitwise-equivalent if, for the same
   input, they produce the same output down to every bit. FP math is
   non-associative, so this reduces to "the FP ops happen in the same order."
2. **Constraint-aware autotuning.** An autotuning *config* (`BLOCK_SIZE`,
   `num_warps`, layout, …) is simultaneously a **performance point** and a
   **numerics point**: it compiles to a particular reduction-tree / accumulation
   order, hence to particular bits. So "stay bitwise-equivalent" turns the
   unconstrained `argmax(perf)` into a **constrained max over a safe set** of
   configs that reproduce a chosen reference order.

The teaching framing: *prune* non-equivalent configs (and genuinely-buggy ones),
*enforce/enlarge* the safe set via a carried order constraint, *optimize* runtime
within it, and *verify* the result. M1 builds the first piece: a static check that
decides, for two compiled kernels, whether their **reductions** are bitwise-equivalent.

"Different bits ≠ incorrect": a different order gives a different *valid* answer.
"Correct" here means *matches the chosen reference bit-for-bit* (determinism vs a
reference), except for genuine bugs.

---

## 2. Architecture & where things live

The reduction-order signature is extracted **MLIR-natively** (since 2026-06-15): a
reusable C++ analysis walks the parsed TTGIR and builds the signature with the
compiler's own layout machinery (`toLinearLayout` + the `tt.reduce` accessors) —
no regex over the IR text. The same analysis is callable from a future in-compiler
verification pass (to assert the order is preserved across passes when the
ordering switch is on).

```
lib/Analysis/ReductionOrder.cpp      — the engine: getReductionOrderSignature(ReduceOp)
include/triton/Analysis/ReductionOrder.h   getReductionOrderSignatures(ModuleOp),
                                           reductionOrdersEquivalent(a,b)  (pass-reusable)
python/src/bitequiv.cc               — pybind: libtriton.bitequiv.reduction_order_signatures
                                           (parses a TTGIR string -> signatures)

bitequiv/
  reduction_tree.py   — thin Python wrapper: parse the TTGIR via a cached MLIRContext
                        and call the C++ analysis. reduction_descriptor / reductions_equivalent.
  equivalence_ttgir.py— autotuner-facing API: reduction_signature (alias), key fns,
                        the CHECKERS registry, and the ir_config_prune predicates
                        (reduction_equivalence_prune / ir_based_prune_configs).
  tests/              — test_reduction_tree.py + test_equivalence_ttgir.py (parse committed
                        real TTGIR fixtures in tests/ttgir/, from gen_ttgir_fixtures.py).
  evaluation/         — GPU evaluation harnesses (static verdict vs torch.equal bits).
  design-doc.md       — this file.   PROGRESS.md — status.   CLAUDE.md — guide.

python/triton/runtime/autotuner.py
                      — the `ir_config_prune` hook (Triton core; checkers are
                        injected from bitequiv, so core never imports bitequiv).
```

Compilation pipeline (where the bits get decided):
`Python → TTIR (layout-free) → TTGIR (layout assigned) → LLVM IR → PTX → (ptxas) → SASS`.
We check at **TTGIR**, because that is the first level where the layout — and hence
the reduction tree — is explicit, yet still above the `ptxas` black box.

---

## 3. Core technical model — the thread↔data map as a *logical relation*

A reduction distributes `S` elements across threads, then combines them. The result
depends on *which thread/lane/warp holds which element* — the **layout**. A naive
encoding is an `O(S)` table `element -> (register, lane, warp)`. We never build that:
the map is **structured** (affine / mixed-radix), so we store its *generators* and
compute coordinates by arithmetic.

For a `#blocked` layout, along the reduce axis `a` the element index `i` decomposes
in mixed radix (innermost-first), with radices read off the encoding:

```
slot  =  i % c            c = sizePerThread[a]   — contiguous elems per thread (registers)
lane  = (i // c) % t       t = threadsPerWarp[a]  — lanes spanning the axis
warp  = (i // (c·t)) % w    w = warpsPerCTA[a]     — warps spanning the axis
group =  i // (c·t·w)       #groups = ceil(S / (c·t·w))  — tile replication (more registers)
```

`order` says which dim is contiguous (its head is the innermost dim), fixing the
per-component strides along the axis. Storage is a handful of integers — `O(polylog S)`.

**General formalism — what we actually use.** Triton represents this exact map as a
`LinearLayout`: a linear map over GF(2) (bit-vectors) from hardware coordinates
`(register, lane, warp, block)` to logical tensor coordinates, stored as `O((log S)²)`
basis vectors, encoding-agnostic (`#blocked`/`#linear`/`#mma`). The mixed-radix
arithmetic above is the `#blocked` special case. The checker computes the real
`LinearLayout` of the reduce operand via `toLinearLayout` and uses its
axis-projected `sublayout` as the signature (§5), so all encodings and
non-contiguous axes are handled by one mechanism — no per-encoding special cases.

Constants at compile time (why this is concrete, not symbolic): the **thread count**
(`num_warps × 32`) and every **tensor shape** are compile-time constants in TTGIR
(Triton tensor shapes are always static `constexpr`s). The runtime problem size `N`
only controls grid/loop counts and masks — it never enters the `tt.reduce` tree, and
is identical across the configs being compared. (The genuine exception is cross-CTA
*atomic* merges, which are nondeterministic and out of scope by construction.)

---

## 4. From layout to reduction tree

The lowering (`lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp`) combines in three
phases, keyed exactly on the radices above:

1. **Within-thread** (`reduceWithinThreads`): each thread folds the `c·#groups`
   elements it holds. `unordered` ⇒ sequential **left fold** in index order;
   `inner_tree` ⇒ a **balanced** pairwise tree (`reduceValueSequence`).
2. **Within-warp** (`warpReduce`): a **butterfly** (XOR) shuffle over the `t` lanes —
   `shfl.sync.bfly.b32`. `unordered` ⇒ **count-down** offsets `t/2,…,2,1`;
   `inner_tree` ⇒ **count-up** `1,2,…,t/2`. A butterfly over `2^k` lanes is a perfect
   binary tree over the `k` lane-index bits, so its order is captured by `(k, bit-direction)`.
3. **Cross-warp**: warp leaders write partials to shared memory; one warp
   butterfly-reduces the `w` partials.

The combine op (`arith.addf`/`mulf`/…) is inlined verbatim from the reduce region.
Two facts make this clean for floats: hardware `redux.sync` is **never** emitted for
float `add`/`mul` (only integers, plus Blackwell f32 min/max), and the shuffle
direction/fold shape are deterministic from the `reduction_ordering` attribute.

`inner_tree` is special: it is **layout-invariant** by construction — it realizes one
fixed balanced tree over the *original* element indices `0…S-1` regardless of layout.
So all `inner_tree` configs with the same `S` reduce identically. (This is why the
enforce mode works: the order no longer depends on the config.)

---

## 5. The equivalence algorithm

### 5.1 The MLIR-native signature (production)

Each `tt.reduce` is summarized as a canonical, hashable **signature string** built
by the C++ analysis (`getReductionOrderSignature`); two TTGIRs are equivalent iff
their signature tuples are equal. The signature is:

`reduce | axis | ordering | nops | combine | layout`

- **`axis`** — `op.getAxis()`.
- **`ordering`** — `getReductionOrderingAttr()`, normalized (null/empty ⇒ `"unordered"`).
- **`combine`** — the ordered op-name sequence of the combine region (a pre-order
  walk). Distinguishes `addf`/`mulf`/`maxnumf`/`cmpf+select` (argmax/argmin) etc.,
  uniformly for single- and multi-operand reduces.
- **`layout`** —
  - `inner_tree`: `"inner_tree-invariant | sAxis=<shape[axis]>"`. The order is
    layout-invariant by construction, so the layout is dropped; only the axis extent
    is kept (a different number of leaves is a different canonical tree).
  - otherwise: the **axis-projected LinearLayout** of the operand,
    `toLinearLayout(srcTy).sublayout({register,lane,warp,block},{dim<axis>})`,
    serialized via `toString()`. This is the real thread↔data map restricted to the
    reduce axis — it fixes how elements are distributed across registers/lanes/warps
    along the axis, hence the reduction tree. `sublayout` keeps the out-dim size, so a
    different axis extent (e.g. a different `BLOCK_SIZE`) yields a different signature
    automatically (shape-soundness, for free).

This replaces the earlier regex-over-text descriptor: we now read the encoding and
compute the layout with the compiler's own `toLinearLayout`, so the signature is
correct across encodings (`#blocked`/`#linear`/`#mma` all normalize through it) and
robust to IR-printing changes.

### 5.2 Soundness (the guarantee) and conservative incompleteness (the price)

`reductions_equivalent(a, b)` ≜ `reduction_descriptor(a) == reduction_descriptor(b)`.

**Soundness:** every fact that can change the tree is in the signature (axis,
ordering, combine, the axis-projected LinearLayout, and — via the LinearLayout
out-dim size — the shape), so equal signatures ⇒ identical trees ⇒ identical bits
*at TTGIR*. The relation **never wrongly merges** two configs whose bits differ — the
kept set is always a safe subset.

**Conservative incompleteness:** the reverse does not hold — some genuinely-equivalent
configs get different signatures (e.g. masked-zero `+0.0` padding that changes the
axis extent; or `inner_tree` reductions whose layouts differ — these are split because
the axis-LL differs, even though the order is invariant). These misses cost tuning
freedom, never correctness, and are **measured, not assumed**, by the GPU evaluation.

**Soundness guard for ops we do not model.** A reduction-like op the analysis cannot
reason about precisely — currently a tensor-core/**MMA** accumulation (`tt.dot` /
`*mma*`, which has no `tt.reduce`) — must **not** collapse to an empty signature, or
two such modules would compare equal (`() == ()`) and be *unsoundly* declared
equivalent (the earlier evaluation caught this as gemm false-positives). So
`getReductionOrderSignatures` appends a conservative `unanalyzed-mma | <fingerprint>`
entry (the dot ops' names + attributes + operand/result types) — such modules match
only when that structure is identical, never on an empty signature. `()` is returned
only when there is no reduction-like op at all. (Multi-operand reduces are no longer
in this bucket: the MLIR analysis handles them via the combine-region key + the
axis-projected LinearLayout, and the evaluation confirms it *detects* Welford
divergence rather than abstaining.)

### 5.3 Validation (GPU evaluation, not an oracle)

The static verdict is validated against **real bits** by `bitequiv/evaluation/`: it
runs a config matrix, compares each pair's static verdict against bit-identical
outputs over many random inputs, and reports a confusion matrix. The pass gate is
**zero soundness false-positives** (a "declared equivalent but bits differ" pair).
Latest run (H100): in-scope kernels sound and exact; multi-operand/MMA boundary
cases sound, with Welford order-divergence detected. (This replaced the earlier
pure-Python canonical-tree oracle — the GPU evaluation is the ground truth now.)

---

## 6. Public interface & usage

The Python surface is unchanged, but it now calls into the C++ analysis, so it
needs the **built triton** (`libtriton.bitequiv`). Parsing a TTGIR string is still
CPU-only — **no GPU, no kernel launch**; only a C++ rebuild is required when the
analysis changes. Import from `bitequiv` (repo root on `sys.path`; namespace package).

### 6.1 Standalone check on two TTGIR modules

```python
from bitequiv.equivalence import reductions_equivalent, reduction_signature, classify

reductions_equivalent(ttgir_a, ttgir_b)      # -> bool : same bitwise reduction order?
reduction_signature(ttgir)                   # -> hashable descriptor tuple (one entry per tt.reduce)
classify({"nw2": g2, "nw4": g4, "nw8": g8})  # -> OrderedDict{signature: [labels...]}  (equivalence classes)
```

`reductions_equivalent` / `reduction_descriptor` live in `bitequiv.reduction_tree`;
`reduction_signature` / `same_reduction_order` in `bitequiv.equivalence` are aliases.

### 6.2 Getting TTGIR without a GPU

Compile-only (no launch) and read the artifact dict:

```python
ck = kernel.warmup(*args, grid=(1,), **constexprs)   # JITFunction.run with warmup=True
ttgir = ck.asm["ttgir"]                               # also "ttir"/"llir"/"ptx"
```

### 6.3 Enabling equivalence pruning in the autotuner

Register an `ir_config_prune` predicate via `prune_configs_by`. The autotuner compiles
every config once (`warmup=True`, no launch), then keeps only configs whose
reduction-order key matches the **reference** (the first config by default):

```python
import triton, triton.language as tl
from bitequiv.equivalence import reduction_equivalence_prune

prune = reduction_equivalence_prune(level="ttgir")   # keep a handle for introspection

@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": 4096}, num_warps=nw) for nw in (2, 4, 8)],
    key=["N"],
    prune_configs_by={"ir_config_prune": prune},
)
@triton.jit
def sum_kernel(src, dst, N, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(src + offs, mask=offs < N, other=0.0)
    tl.store(dst, tl.sum(x, axis=0))

# after a tuning run:
prune.classes   # {signature: [Config, ...]}  — the equivalence classes seen
prune.pruned    # {Config: "not-equivalent-to-reference"}  — what was dropped
```

- **Levels:** `level="ttgir"` is the only level on this branch; a list of levels is also
  accepted (keys combined; kept iff equal at every level). The **PTX backstop**
  (`level="ptx"` / `"both"`) is implemented on the sibling `bitequiv-m1-ptx` branch
  (§6.5). Levels resolve through the injected `CHECKERS` registry, so Triton core stays
  decoupled from `bitequiv`.
- **Reference selection** (`reduction_equivalence_prune(level, reference=...)`):
  `None` = the first compiled config (autotuner-supplied); or an **external anchor not
  in the tuning set** — a pre-compiled kernel (`.asm`), an `asm` dict, or raw IR text —
  whose key fixes the order to match (e.g. enforce a golden config or cuBLAS later).

### 6.4 Extending the checker (key functions / registry)

A *key function* maps a compiled artifact to a hashable equivalence key:
`key_fn(config, asm, metadata) -> Hashable` (configs with equal keys are equivalent).
`reduction_equivalence_key` is the TTGIR one (`reduction_signature(asm["ttgir"])`).
Wrap any key fn into a predicate with `ir_based_prune_configs(key_fn, reference=...)`.
New levels are added to the `CHECKERS` registry (this branch ships `{"ttgir": ...}`; the
`"ptx"` entry is added on the `bitequiv-m1-ptx` branch).

### 6.5 The PTX backstop (FMA contraction)

> **Implemented on the sibling `bitequiv-m1-ptx` branch** (independent split). The design
> below is retained here as the whole-project reference; the files it names
> (`bitequiv/ptx_reduction.py`, `equivalence_ptx.py`, `tests/fixtures/ptx/`) live on that
> branch, not this one.

TTGIR fixes the *association order* but is **provably blind to FMA contraction**:
whether a `mul` feeding a reduction fuses with the following `add` into one rounded
`fma` — or stays as two separately-rounded ops — is decided *below* TTGIR, in the
LLVM NVPTX backend / `ptxas` (gated by `enable_fp_fusion` / `--fmad`). Fused vs
unfused gives different bits. Two configs can therefore compile to **byte-identical
TTGIR yet bit-different PTX**. Verified: a `tl.sum(x*y)` dot reduction emits
`fma.rn.f32` with fusion on and `add.rn.f32` + `mul.rn.f32` with fusion off — same
TTGIR, different PTX (`bitequiv/tests/fixtures/ptx/dot_fuse_{on,off}.*`).

`ptx_reduction_signature` (engine: `bitequiv/ptx_reduction.py`) closes this gap by
reconstructing a signature from the PTX of each `.entry`:

- **ordered butterfly-shuffle offsets** — each warp / cross-warp step is a
  `shfl.sync.bfly.b32 dst, src, OFFSET, …`; the *ordered* offset sequence encodes the
  within-warp tree (`16,8,4,2,1` count-down for `unordered`, `1,2,…,16` count-up for
  `inner_tree`) plus the appended cross-warp tree, whose length grows `log2(num_warps)`
  — so one sequence captures both warp shape and warp count;
- **fp combine opcodes with full modifiers** — the multiset of
  `{add,mul,fma,…}.<mods>.f{16,32,64}` tokens; the modifiers *are* the signal
  (`fma.rn.f32` vs `add.rn.f32`+`mul.rn.f32` = fusion; `.rn` = rounding; `.ftz` =
  flush-to-zero), and the count tracks the within-thread fold length;
- a derived **fused** flag (any `fma.*`) — an explicit, redundant readout of the
  TTGIR-invisible decision.

Same soundness contract as TTGIR (equal signature ⇒ identical PTX reduction ⇒
identical bits; conservative, may over-split). The **refinement invariant** here is
*PTX refines TTGIR*: on pure-add reductions the PTX and TTGIR signatures partition the
configs identically (test: `test_fixture_ptx_partition_matches_ttgir_on_pure_sum`);
on the dot reduction PTX splits the fp-fusion pair that TTGIR merges. Single-reduction
scope (per `.entry`), matching the TTGIR checker — pair it with TTGIR via
`reduction_equivalence_prune("both")` (kept iff equal at *both* levels).

**Residual caveat.** PTX sits *above* the `ptxas` → SASS gap; `ptxas` can still
contract/reorder. PTX is the practical backstop, not absolute ground truth (SASS via
`cuobjdump` would be); `--fmad=false` pins the fusion decision.

---

## 7. Scope, limitations & roadmap

**Validated sound on the GPU** (`bitequiv/evaluation/`, H100): single- and
multi-operand `tt.reduce` (sum/prod/max/min/argmin/Welford) over distributed
encodings, any axis (1-D / 2-D), including looped/large-N and bf16/fp16. Because the
signature is the real axis-projected `LinearLayout`, encodings (`#blocked`/`#linear`/
`#mma`) and non-contiguous axes are handled uniformly; multi-operand reduces are now
*analyzed* (Welford divergence is detected, not just abstained).

**Conservative / future (sound today, not yet precise):**
- **MMA / `tl.dot`** (the K-axis accumulation): no `tt.reduce` to parse, so it is
  covered by the §5.2 conservative `unanalyzed-mma` guard (sound — precision modes
  are detected, tiling is over-split). Precise MMA accumulation-order modeling is the
  M3 GEMM milestone.
- **Masking / identity padding**: a config that over-covers `N` pads with the combine
  identity (`+0.0`), a bitwise no-op, but changes the axis extent → conservatively
  split. Modeling the identity would merge padded vs exact.
- **PTX / FMA backstop** *(done on the `bitequiv-m1-ptx` branch — see §6.5)*:
  `ptx_reduction_signature` reconstructs the shuffle-offset + fp-opcode signature from
  PTX, catching the `mul`+`add → fma` fusion TTGIR cannot see; `level="ptx"` / `"both"`
  are live and GPU-validated there (closes 20 TTGIR false-positive pairs on a dot
  reduction). *Remaining:* the residual `ptxas → SASS` gap (SASS ground truth via
  `cuobjdump`), and multi-reduction entries (single-reduce scope today).
- **In-compiler verification pass**: `reductionOrdersEquivalent` (C++) is ready for a
  pass that asserts the order is preserved across passes when the switch is on.

---

## 8. Rules for future development (follow these)

1. **Soundness is non-negotiable.** Correctness gates performance: a relation must
   **never** declare two configs equivalent when their bits could differ. When unsure,
   be conservative (over-split), never optimistic. Re-run the equivalence tests after
   any perf- or codegen-affecting change.
2. **Conservative-first, with the GPU evaluation as ground truth.** Ship the safe
   relation; measure incompleteness empirically with `bitequiv/evaluation/`
   (static verdict vs `torch.equal` bits). The pass gate is zero soundness
   false-positives.
3. **Keep Triton core decoupled.** Checkers are *injected* (the `CHECKERS` registry /
   `ir_config_prune` predicates); never make `python/triton/...` import `bitequiv`.
4. **MLIR-native, not regex.** The signature is computed in C++ from the parsed IR
   via `toLinearLayout` (the compiler's own layout machinery). Extend the C++ analysis
   (`lib/Analysis/ReductionOrder.cpp`); a rebuild is needed when it changes. The same
   analysis is reused by the future in-compiler verification pass.
5. **Tests are the contract.** Real committed TTGIR fixtures (`tests/ttgir/`, parsed
   CPU-only via `gen_ttgir_fixtures.py`); add a fixture for every new shape/case;
   validate against real bits with the evaluation when a GPU is available. Run `ruff`
   + `yapf` on Python and `clang-format` on C++ (or `pre-commit`) before done.
6. **Keep this doc in sync.** Any change to the model, algorithm, interface, or scope
   updates the relevant section here, and `PROGRESS.md` gets a one-line entry.

---

## 9. Code & lowering anchors

- Engine (C++, MLIR-native): `include/triton/Analysis/ReductionOrder.h` +
  `lib/Analysis/ReductionOrder.cpp` (`getReductionOrderSignature(ReduceOp)`,
  `getReductionOrderSignatures(ModuleOp)`, `reductionOrdersEquivalent`); pybind in
  `python/src/bitequiv.cc` (`libtriton.bitequiv.reduction_order_signatures`).
- Python wrapper (TTGIR): `bitequiv/reduction_tree.py` (`reduction_descriptor`,
  `reductions_equivalent`; cached MLIRContext + the binding).
- Engine (PTX backstop): `bitequiv/ptx_reduction.py` (`ptx_reduction_descriptor`,
  `ptx_reductions_equivalent`; parses `shfl.sync.bfly` offsets + fp-opcode multiset)
  — **on the `bitequiv-m1-ptx` branch.**
- API: `bitequiv/equivalence_ttgir.py` (`reduction_signature`, `same_reduction_order`,
  `reduction_equivalence_key`, `CHECKERS`, `ir_based_prune_configs`,
  `reduction_equivalence_prune`, `classify`). The PTX-level API (`equivalence_ptx.py`,
  `ptx_reduction_signature`) is on the `bitequiv-m1-ptx` branch.
- Autotuner hook: `python/triton/runtime/autotuner.py` (`ir_config_prune`, called
  `(config, asm, metadata, reference)` after a `warmup=True` compile per config) — on
  the `bitequiv-starter` branch.
- Tests: `bitequiv/tests/test_reduction_tree.py`, `bitequiv/tests/test_equivalence_ttgir.py`
  (PTX tests `test_ptx_reduction.py` + `tests/fixtures/ptx/` are on the `bitequiv-m1-ptx`
  branch).
- Lowering: `lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp` (`reduceWithinThreads`,
  `warpReduce` count-down/up, `reduceValueSequence`, `isInnerTree`),
  `ReduceScanCommon.h` (`applyCombineOp`); shuffle `third_party/nvidia/.../Utility.cpp`
  (`shuffleXor` → `shfl.sync.bfly.b32`); `redux.sync` gate `.../TargetInfo.cpp`
  (`matchReduxKind`).
- Layout math (used by the analysis): `include/triton/Tools/LinearLayout.h`
  (`sublayout`, `toString`, `operator==`); `LinearLayoutConversions.h` (`toLinearLayout`,
  `register/lane/warp/block` → `dim{i}` conventions); `lib/Dialect/Triton/IR/Ops.cpp`
  (`ReduceOp::getSingleCombiner` / `hasDefinedOrdering`).
```
