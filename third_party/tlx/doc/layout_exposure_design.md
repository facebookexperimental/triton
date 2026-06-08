# Exposing Shape:Stride layouts on `local_alloc` (QK)

## Motivation
For hand-tuned Blackwell attention kernels, the dominant source of
`convert_layout` / SMEM round-trips is the **register layout the compiler picks
for the QK `tmem_load`**. The compiler's default doesn't match what the
downstream split / per-32-block quantize / scaled-MMA want, so it patches with
converts.

These kernels are authored by people who already think in **Shape:Stride layouts**
(CUTLASS). Rather than have the compiler *infer* the right layout, let the author
**state it directly on the buffer** — the `local_alloc` — so the producing MMA
and every load inherit one canonical layout:

```python
qk_tiles = tlx.local_alloc((BLOCK_N1, BLOCK_M1), tl.float32, NUM_BUFFERS,
                           tlx.storage_kind.tmem,
                           layout=tlx.layout(...))   # FA4 separable TV layout
qk = tlx.local_load(qk_tiles)                              # inherits the alloc layout
```

Explicit author control is the better fit for hand-tuned code, and the buffer is
the right home (see next section): one declaration drives producer + all
consumers via existing propagation.

## Why this is cheap: Triton layouts ARE GF(2)
`ttg.LinearEncodingAttr` (`#linear`) is a `(register, lane, warp, block) →
tensor-index` map with **power-of-2 bases** — i.e. a Shape:Stride **TV (thread-value)
layout** restricted to GF(2), with `thread = (lane, warp)` and `value =
register`. The FA4 separable QK layout I hand-wrote is exactly this:

    #linear17 = register=[[0,1],[0,2],[0,4],[0,8],[0,16],[0,64]]   // value -> K (m), stage in high reg bit
               lane=[[1,0],[2,0],[4,0],[8,0],[16,0]]               // lane  -> N (n)
               warp=[[32,0],[64,0],[0,32]]                          // warp  -> N, then a K block

So "expose Shape:Stride" = give the author an ergonomic way to write a TV layout that
lowers to `LinearEncodingAttr`. Everything downstream already exists.

## Where the layout lives: `local_alloc` (primary), `local_load` (override)
The layout is pinned on the **buffer**, not each access:

- **Single source of truth for producer + all consumers.** The QK buffer is an
  **MMA accumulator** — the MMA *writes* it (TMEM) and `local_load` *reads* it;
  the two views must agree. A `local_load`-only anchor covers just the read side.
  The alloc is the one site both the MMA and every load see, so the layout is
  declared once and checked against both.
- **Multiple loads stay consistent** — full load + subtile loads of the same
  buffer all derive from one declaration instead of risking disagreement +
  converts.
- **Matches the tensor-has-a-layout model** — a tensor *has* a layout. Precedent exists:
  `require_tmem_layout` already attaches a TMEM layout to a memdesc; this
  generalizes it to the TV/register layout.

Two layouts are in play and, for TMEM, are linked:
1. **Memory layout** — the buffer encoding (`#tmem` blockM/blockN/colStride for
   TMEM; `#shared` swizzle for SMEM).
2. **Register/TV layout** — the `#linear` distribution of a *loaded* value (the
   FA4 separable `#linear17`).

`#linear17` is a *register* layout, but for TMEM the achievable load
register-layout is fixed by the TMEM encoding + the `tcgen05.ld` pattern
(x32×2 vs x64). So "layout on `local_alloc`" means: the alloc carries the
canonical TV/Shape:Stride layout, the TMEM organization is chosen to realize it, and
`local_load` **inherits** it (no `layout=` on the load).

`local_load(layout=…)` remains as a **local override** for the rare case where a
second consumer wants a different register layout (it then pays a convert).

## Infrastructure: mostly reused, one new requirement
Reused:
- `LinearEncodingAttr` — the lowering target.
- `builder.make_linear_encoding_attr(regBases, laneBases, warpBases, shape)` —
  **already added + validated** this session; builds `#linear` from bases.
- `tlx.require_layout` + `tlx-propagate-layout` (`RequireLayoutPattern`) — anchor
  the (inherited) load-result layout and propagate it backward, eliminating
  converts.
- `tlx.layout(shape=(thread, value), stride=(thread, value))` — the user-facing
  Shape:Stride spec (`types.py`); a CuTe thread-value layout written **only** in
  shape/stride (no register/lane/warp on the surface). The compiler decomposes
  the modes, splits thread bits into lane/warp + value bits into registers, and
  emits `#linear` via `make_linear_encoding_attr`.

New requirement (the cost of moving the pin to the alloc): **`TMEMAllocation`
and the MMA accumulator-layout choice must respect the alloc's declared layout**
instead of choosing their own. With a `local_load`-only anchor only the read side
is constrained; pinning the buffer also constrains the *producer* MMA, which is
what removes the root mismatch (and must be enforced/verified, not silently
overridden).

## API design — the surface is Shape:Stride, never linear-layout
Users write a **Shape:Stride layout** (`Shape:Stride`, hierarchical). They never see, type,
or reason about `register/lane/warp` bases — `LinearEncodingAttr` is purely the
internal target the compiler maps to (lossless, see the GF(2) equivalence above).

```python
qk_tiles = tlx.local_alloc((BLOCK_N1, BLOCK_M1), tl.float32, NUM_BUFFERS,
                           tlx.storage_kind.tmem,
                           layout=tlx.layout("(T, V) : (...)"))   # thread-value layout
qk = tlx.local_load(qk_tiles)   # inherits; Shape:Stride is mapped to #linear under the hood
```

- `tlx.layout("…")` (and/or a structured Shape:Stride builder) is the **only**
  user-facing layout object.
- `tlx.local_alloc(..., layout=l)` records it as the buffer's canonical layout;
  `local_load` inherits; the producing MMA accumulator is constrained to match.
- A **named-preset** layer can wrap common Shape:Stride layouts so authors
  don't hand-write them.
- `local_load(buf, layout=l2)` is the per-access override.

## Shape:Stride → LinearLayout (the core compiler component)
This translation is now the **central new piece**, not optional sugar. Steps:

1. **Parse** the `Shape:Stride` (nested/hierarchical) into flat modes, each
   `(extent, stride)`.
2. **Power-of-2 decompose** each mode: a mode of extent `2^k`, stride `s` →
   `k` basis vectors `s, 2s, …, 2^{k-1}s`, each mapped onto the output (tensor)
   dimension via the stride. **Constraint:** extents/strides must be powers of
   two (LinearLayout is GF(2)); GPU TV layouts incl. FA4's satisfy this — reject
   non-pow2 with a clear error. This GF(2) decomposition is exactly why the map
   is lossless (LinearLayout = XOR).
3. **Assign modes to `register` / `lane` / `warp`.** This is the one real design
   decision: a bare Shape:Stride layout doesn't say which modes are value vs thread, or
   how thread splits across lane/warp. Pick a convention:
   - The user's Shape:Stride layout is a **TV layout** `(T, V) → coord`, where `T` is the
     thread id `0..num_threads-1` and `V` the value/register id.
   - The translator splits `T` → `lane` (low `log2(threadsPerWarp)` bits) +
     `warp` (the rest), and `V` → `register`.
   - (Alternative: let the author tag modes as `@lane/@warp/@reg`. The TV
     convention is more idiomatic and matches CUTLASS MMA atoms.)
4. **Emit** `LinearEncodingAttr` via the existing
   `make_linear_encoding_attr(regBases, laneBases, warpBases, shape)` — internal
   only; the user never touches it.

A **named-preset** path can bypass parsing and emit a common layout directly
(e.g. `tlx.separable_layout(axis=K, vec=32)`), built from the same
`make_linear_encoding_attr`.

Where it runs: a frontend/builder utility (concrete layouts).

## num_warps
An *explicit* author layout is **concrete** — no placeholder/deferred-resolve
machinery needed. If the author wants it written parametrically in `num_warps`
(so it follows an `async_task`'s effective warp count after inlining), a
deferred placeholder + resolve pass would be needed (not implemented).
Default: concrete.

## What this gets you automatically vs. what still needs a source change
Pinning **only** the QK buffer (`local_alloc`) to the separable layout:

| Result | Auto from the QK anchor? | Why |
|---|---|---|
| `888afb61e` qk/P in `#linear17` | **yes** | elementwise `fma/exp2` forward-propagate the anchor |
| `_split_n_2D(pT)` becomes free | **yes** | stage is a register bit → reshape/trans/split is a register relabel, not SMEM |
| one `exp2`, no recompute | **yes** | single P feeds the free split; nothing forces a reload |
| dP/dS in `#linear16` | **likely** | `dS=pT·(dP−Di)` unifies layouts, pushes `#linear16` back onto the flexible `dpT` load |
| P f8-store / dS-quant convert-free | **yes** | `#linear17/16` are store/pack compatible |
| `82af0be2b` kill dQ `dsT_t` transpose | **NO** | structural rewrite: store dS non-transposed + `memdesc_trans` at the MMA, justified by the **scalar** dQ scale. Layout propagation only removes `convert_layout`, not an explicit `tt.trans`. |

So exposing the QK layout ≈ delivers **`888`** (the separable unification) through
existing propagation, with two propagation-dependent caveats (the `dpT` re-anchor
and the reshape/trans being chosen in free form) and the residual tiny scale
converts.

`82af` is **out of scope** for layout exposure: it must come from the source
(`tlx.local_trans(ds_dq)` + a non-transposed `ds_dq` alloc) — exactly what the
sibling commit `7467df14f` does in the Python kernel. It's a DSL/op change, not
a layout one.

## Implementation plan
1. **Phase 1 — Shape:Stride→LinearLayout + `local_load(layout=)`** — DONE:
   `tlx.layout(shape=(thread, value), stride=(thread, value))` (`types.py`), a
   shape/stride-only TV layout, decomposes the GF(2) modes (splitting thread →
   lane/warp, value → register internally) and builds `#linear` via
   `make_linear_encoding_attr`; `tlx.local_load(buf, layout=…)` pins it (TMEM:
   load-result encoding; SMEM: `require_layout`). Validated `tlx.layout(...) ==
   #linear17`.
2. **Phase 1a — `local_alloc(load_layout=)`** — DONE: carry the layout on the
   buffer (separate from the memory `layout=`) so every `local_load` (through
   `local_view`) inherits it without a per-load `layout=`. Validated that the
   inherited load produces the same `#linear` as an explicit `local_load(layout=)`
   (unit test `test_local_alloc_load_layout_compile`).
3. **Phase 1b — producer side:** make `TMEMAllocation` / the MMA accumulator
   layout respect the alloc's declared layout (verify, don't override).
4. **Phase 2 — ergonomics:** richer layout surface (hierarchical/nested, named
   presets library); keep `local_load(layout=…)` as the per-access override.
4. **Phase 3 — parametric (num_warps-aware) layouts** via the existing
   placeholder rails, if needed.
5. **Tests:**
   - unit: Shape:Stride string → `LinearEncodingAttr`, asserting the FA4 layout maps to
     `#linear17` / `#linear16` (we already have the `==` check for the bases).
   - lit: a `local_alloc(layout=l)` loaded → `tlx-propagate-layout` → CHECK
     the load adopts it and `convert_layout` is gone; a second test that the
     producer MMA accumulator matches.

## Open questions
- **layout surface scope:** which Shape:Stride subset (flat `Shape:Stride`, nested,
  composed/`logical_divide`)? Start with flat power-of-2 TV layouts.
- **Mode→partition convention:** TV layout `(T,V)` with `T` split into
  lane/warp (CUTLASS-idiomatic), or explicit `@lane/@warp/@reg` mode tags?
- Does `local_alloc(layout=l)` carry the **register/TV** layout (TMEM encoding
  derived to realize it), or imply the **TMEM memory encoding** directly? For
  TMEM these are linked; pick the one authors find natural.
- How strictly to enforce alloc-vs-MMA-accumulator agreement — hard verify error,
  or auto-insert a producer-side conversion?
- Preset library: which named Shape:Stride layouts (FA4 separable, x32×2, …) to expose.
