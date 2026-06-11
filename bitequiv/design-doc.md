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

Status: M1 first cut (2026-06-11). TTGIR-level, single-operand `#blocked` reductions.
PTX/FMA backstop, multi-operand reduces, and `#mma`/`#linear` are future work (§7).

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

```
bitequiv/
  reduction_tree.py   — logical-relation core: the thread<->data descriptor (production)
                        + the canonical-tree oracle (tests). The engine.
  equivalence.py      — autotuner-facing API: reduction_signature (alias), key fns,
                        the CHECKERS registry, and the ir_config_prune predicates
                        (reduction_equivalence_prune / ir_based_prune_configs).
  tests/              — test_reduction_tree.py (descriptor + oracle),
                        test_equivalence.py (API + autotuner reference-prune).
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

**General formalism (the endgame).** Triton already represents this exact map as a
`LinearLayout`: a linear map over GF(2) (bit-vectors) from hardware coordinates
`(register, lane, warp, block)` to logical tensor coordinates, stored as `O((log S)²)`
basis vectors, encoding-agnostic (`#blocked`/`#linear`/`#mma`). The mixed-radix
arithmetic above is the `#blocked` special case. The first cut re-derives the
`#blocked` case in Python (no C++ rebuild); the natural upgrade computes the leaf map
via `toLinearLayout` + `ReduceOpHelper` so non-`#blocked` encodings work too (§7).

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

### 5.1 Conservative descriptor (production)

Each `tt.reduce` is summarized as a small hashable **descriptor**; two TTGIRs are
equivalent iff their descriptor tuples are equal. Per reduce:
`(axis, ordering, combine, layout_term)` where

| `ordering` | `layout_term` |
|---|---|
| `inner_tree` | `("inner_tree-invariant", S_axis)` — layout dropped; `S` kept |
| `#blocked`   | `("blocked", S_axis, sizePerThread[a], threadsPerWarp[a], warpsPerCTA[a], order)` |
| other enc.   | `("raw", normalized_encoding_text, S_axis)` — verbatim exact-match |

- `combine` = the sorted unique `arith.*`/`math.*` ops in the region.
- `ordering` is normalized: absent/empty ⇒ the default, recorded as `"unordered"`.
- **`S_axis` (operand extent on the reduce axis) is the key field.** The number of
  within-thread register groups is `ceil(S / (c·t·w))`, so two configs with identical
  `#blocked` params but a different axis extent (e.g. a different `BLOCK_SIZE`) fold a
  different number of elements per thread → a different tree → different bits. Omitting
  `S` is **unsound** (it would merge them); it matters for `inner_tree` too.

### 5.2 Soundness (the guarantee) and conservative incompleteness (the price)

`reductions_equivalent(a, b)` ≜ `reduction_descriptor(a) == reduction_descriptor(b)`.

**Soundness:** every fact that can change the tree is in the descriptor, so equal
descriptors ⇒ identical trees ⇒ identical bits *at TTGIR*. The relation **never
wrongly merges** two configs whose bits differ — the kept set is always a safe subset.

**Conservative incompleteness:** the reverse does not hold — some genuinely-equivalent
configs get different descriptors:
- *Cross-level re-association.* `t=2, w=1` (a 2-lane shuffle) and `t=1, w=2` (a 2-warp
  smem combine) both compute one `a+b`, identical bits, but their `(t,w)` differ.
- *Masking / identity padding.* Over-covering `N` pads the tail with the combine's
  identity (`+0.0`), a bitwise no-op for add, but changes `S`.
- *Different encodings, same map* (`#blocked` vs equivalent `#linear`) — the `"raw"`
  branch compares text.

These misses cost tuning freedom, never correctness. They are acceptable for the first
cut and **measured, not assumed**, by the oracle.

### 5.3 Canonical-tree oracle (tests, ground truth)

To know what the conservative relation misses, tests build the **actual
parenthesization over element indices** and canonicalize it (`oracle_tree`): replay
the three phases over `0…S-1`, then sort commutative children, keep binary grouping,
and elide trivial 1-child levels (so phase origin is erased — a 2-lane merge and a
2-warp merge become the same node). Tests assert the

> **refinement invariant:** descriptor-equal ⇒ oracle-equal

over a parameter matrix (a guard that no descriptor field is ever dropped), and record
a concrete *oracle-equal ∧ descriptor-different* case so the gap stays visible. The
oracle is `O(S)` and lives only in the test path — production never materializes a tree.

---

## 6. Public interface & usage

Everything is pure-Python and needs **no GPU and no C++ rebuild**. Import from
`bitequiv` (the repo root must be on `sys.path`; `bitequiv` is a namespace package).

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

- **Levels:** `level="ttgir"` (works today), `"ptx"` (stub — raises `NotImplementedError`
  until the backstop is built), `"both"`, or a list (keys combined; kept iff equal at
  every level). Levels resolve through the injected `CHECKERS` registry, so Triton core
  stays decoupled from `bitequiv`.
- **Reference selection** (`reduction_equivalence_prune(level, reference=...)`):
  `None` = the first compiled config (autotuner-supplied); or an **external anchor not
  in the tuning set** — a pre-compiled kernel (`.asm`), an `asm` dict, or raw IR text —
  whose key fixes the order to match (e.g. enforce a golden config or cuBLAS later).

### 6.4 Extending the checker (key functions / registry)

A *key function* maps a compiled artifact to a hashable equivalence key:
`key_fn(config, asm, metadata) -> Hashable` (configs with equal keys are equivalent).
`reduction_equivalence_key` is the TTGIR one (`reduction_signature(asm["ttgir"])`).
Wrap any key fn into a predicate with `ir_based_prune_configs(key_fn, reference=...)`.
New levels are added to `CHECKERS = {"ttgir": ..., "ptx": ...}`.

---

## 7. Scope, limitations & roadmap

**In scope (first cut):** single-operand `tt.reduce` (sum/prod/max/min) over a
`#blocked` operand, any axis (1-D / 2-D).

**Out of scope / conservative today, in roadmap order:**
- **Multi-operand reduces** (argmax/argmin/Welford): the parser matches a single
  operand; multi-operand ops `(tensor<…>, tensor<…>)` with a `2N`-arg region don't
  match yet (no false merge). *Next.*
- **Non-contiguous reduce axis**: descriptor stays sound (includes `order`); the oracle
  assumes a contiguous axis — generalize the leaf map for exact modeling.
- **`#mma` / `#linear`**: compared verbatim today. Upgrade the leaf map to the real
  `LinearLayout` (`toLinearLayout` + `ReduceOpHelper`) — encoding-agnostic, and
  recognizes cross-encoding equivalence (M3/GEMM needs it).
- **Masking / identity padding**: model the combine's identity to merge padded vs exact.
- **PTX / FMA backstop**: TTGIR is blind to `mul`+`add → fma` fusion (decided below
  TTGIR by `enable_fp_fusion` / `ptxas --fmad`). Implement `ptx_reduction_signature`
  (the only change needed to light up `level="ptx"`).
- **Validate against real bits**: an experiment harness that runs a config matrix and
  `torch.equal`s outputs, confirming the static verdict (the guardrail).

---

## 8. Rules for future development (follow these)

1. **Soundness is non-negotiable.** Correctness gates performance: a relation must
   **never** declare two configs equivalent when their bits could differ. When unsure,
   be conservative (over-split), never optimistic. Re-run the equivalence tests after
   any perf- or codegen-affecting change.
2. **Conservative-first, with the oracle as ground truth.** Ship the safe relation;
   measure incompleteness with the canonical-tree oracle. The **refinement invariant**
   (descriptor-equal ⇒ oracle-equal) must hold in tests for every new descriptor field.
3. **Keep Triton core decoupled.** Checkers are *injected* (the `CHECKERS` registry /
   `ir_config_prune` predicates); never make `python/triton/...` import `bitequiv`.
4. **Pure-Python, no rebuild** for the checker (offline devgpu constraint). The
   descriptor abstraction is designed so the leaf-map source can later swap to the real
   `LinearLayout` *without changing the public interface*.
5. **Tests are the contract.** Pure-string TTGIR fixtures (no GPU); add a fixture for
   every new shape/case; validate against real bits via `torch.equal` when a GPU is
   available. Run `ruff` + `yapf` (or `pre-commit`) before done.
6. **Keep this doc in sync.** Any change to the model, algorithm, interface, or scope
   updates the relevant section here, and `PROGRESS.md` gets a one-line entry.

---

## 9. Code & lowering anchors

- Engine: `bitequiv/reduction_tree.py` (`reduction_descriptor`, `reductions_equivalent`;
  oracle `oracle_tree` / `oracle_tree_from_blocked` / `trees_equivalent`).
- API: `bitequiv/equivalence.py` (`reduction_signature`, `same_reduction_order`,
  `reduction_equivalence_key`, `CHECKERS`, `ir_based_prune_configs`,
  `reduction_equivalence_prune`, `ptx_reduction_signature` stub, `classify`).
- Autotuner hook: `python/triton/runtime/autotuner.py` (`ir_config_prune`, called
  `(config, asm, metadata, reference)` after a `warmup=True` compile per config).
- Tests: `bitequiv/tests/test_reduction_tree.py`, `bitequiv/tests/test_equivalence.py`.
- Lowering: `lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp` (`reduceWithinThreads`,
  `warpReduce` count-down/up, `reduceValueSequence`, `isInnerTree`),
  `ReduceScanCommon.h` (`applyCombineOp`); shuffle `third_party/nvidia/.../Utility.cpp`
  (`shuffleXor` → `shfl.sync.bfly.b32`); `redux.sync` gate `.../TargetInfo.cpp`
  (`matchReduxKind`).
- Layout math (GF(2) endgame): `include/triton/Tools/LinearLayout.h`;
  `include/triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h` (`toLinearLayout`);
  `include/triton/Analysis/Utility.h` (`ReduceOpHelper`).
```
