# Reduction Ordering in Triton

## Problem

Triton's default reduction (`tl.sum`, `tl.reduce`) uses a layout-dependent
accumulation order. The compiler maps tensor elements to threads based on the
chosen encoding (number of warps, block size, etc.) and reduces in whatever
order falls out of that mapping. This means changing `num_warps` or
`BLOCK_SIZE` can change the floating-point result, because floating-point
addition is not associative.

For workloads that require **bitwise reproducibility** — deterministic training,
numerical debugging, regression testing — a layout-independent reduction order
is necessary.

## Solution: `reduction_ordering` Parameter

The `reduction_ordering` parameter on `tl.sum` and `tl.reduce` lets the user
request a specific, deterministic accumulation order that is independent of
the thread layout. The system guarantees that, given the same logical input
data and reduction ordering, the result is bitwise identical regardless of
`num_warps`, memory layout (row-major vs column-major), or other compilation
parameters.

### Usage

```python
# Sum with deterministic ordering
z = tl.sum(x, axis=1, reduction_ordering=tl.ReductionOrdering.INNER_TREE)

# Custom combine function with deterministic ordering
z = tl.reduce(x, axis=1, combine_fn=my_fn,
              reduction_ordering=tl.ReductionOrdering.INNER_TREE)

# Default (no ordering guarantee, best performance)
z = tl.sum(x, axis=1)  # equivalent to ReductionOrdering.UNORDERED
```

Because `ReductionOrdering` objects cannot be used directly inside JIT-compiled
code (they are Python objects without a Triton type), pass them as `tl.constexpr`
kernel parameters:

```python
@triton.jit
def kernel(X, Z, ORDERING: tl.constexpr):
    x = tl.load(X + tl.arange(0, 1024))
    z = tl.sum(x, axis=0, reduction_ordering=ORDERING)
    tl.store(Z, z)

kernel[(1,)](x, z, ORDERING=tl.ReductionOrdering.INNER_TREE, num_warps=4)
```

---

## Architecture

### Data Flow

```
Python user code
  tl.sum(x, axis=1, reduction_ordering=tl.ReductionOrdering.INNER_TREE)
    │
    ▼
core.py: reduce()           — validates type, defaults None → UNORDERED
    │  passes ordering.name string ("inner_tree", "unordered", or "")
    ▼
semantic.py: reduction()     — calls builder.create_reduce(..., reduction_ordering="inner_tree")
    │
    ▼
ir.cc: create_reduce         — sets StringAttr "reduction_ordering" on ReduceOp
    │
    ▼  [TTIR → TTGIR: attribute preserved via addNamedAttrs]
    ▼
Utility.cpp                  — getNumContiguousGroupsOnAxis() reads attr, computes K
    │
    ▼
ReduceOpToLLVM.cpp           — isInnerTree() checks attr; modifies all 6 reduction phases
    │
    ▼
LLVM IR / PTX                — deterministic shuffle order baked into generated code
```

### Key Concept: The `reduction_ordering` Attribute

The ordering is a **named attribute** (not a formal ODS attribute) set via
`op->setAttr()` on the `ReduceOp`. It is a `StringAttr` with values:

- `"inner_tree"` — deterministic inner-tree ordering
- `"unordered"` or absent — default layout-dependent ordering

The attribute automatically survives TTIR → TTGIR lowering because
`addNamedAttrs` copies all named attributes from the source op.

---

## Frontend (Python)

### Type Hierarchy

**File: `python/triton/language/core.py`, lines 25–86**

```
ReductionOrderingBase (abstract base)
  ├── ReductionOrdering         — a named strategy ("inner_tree", "unordered")
  └── CompositeReductionOrdering — chains strategies (not yet implemented)
```

- **`ReductionOrdering`**: Has a `name` field. Two predefined constants:
  - `ReductionOrdering.UNORDERED` — default, no ordering guarantee
  - `ReductionOrdering.INNER_TREE` — deterministic tree-based ordering

- **`CompositeReductionOrdering`**: Forward-looking extensibility for composing
  orderings across different levels of the reduction tree (e.g., within-thread
  vs across-warp). Currently raises `TypeError` if used.

### Validation

**File: `python/triton/language/core.py`, `reduce()` function (~line 2725)**

- `None` defaults to `ReductionOrdering.UNORDERED`
- `CompositeReductionOrdering` raises `TypeError`
- Non-`ReductionOrdering` types raise `TypeError`

### Plumbing to C++

**File: `python/triton/language/semantic.py`, `reduction()` method (~line 1890)**

Passes `reduction_ordering.name` (a string like `"inner_tree"`) to
`builder.create_reduce()`.

**File: `python/src/ir.cc`, `create_reduce` binding (~line 1776)**

Sets `StringAttr` on the MLIR `ReduceOp`:
```cpp
reduceOp->setAttr("reduction_ordering",
    StringAttr::get(reduceOp->getContext(), reductionOrdering));
```

---

## Backend (C++)

### Analysis: Contiguous Groups

**File: `lib/Analysis/Utility.cpp`, `getNumContiguousGroupsOnAxis()` (~line 110)**

```cpp
unsigned ReduceOpHelper::getNumContiguousGroupsOnAxis() {
  auto reductionOrderingAttr =
      op->getAttrOfType<StringAttr>("reduction_ordering");
  if (!reductionOrderingAttr ||
      reductionOrderingAttr.getValue() != "inner_tree")
    return 1;
  unsigned elemsPerThread = triton::gpu::getElemsPerThread(srcTy)[axis];
  unsigned contigPerThread = triton::gpu::getContigPerThread(srcTy)[axis];
  return elemsPerThread / contigPerThread;
}
```

**K** (the return value) is the number of contiguous groups each thread holds
along the reduction axis. For the default ordering, K=1 (everything is treated
as one group). For inner tree, K = `elemsPerThread / contigPerThread` — each
contiguous run of elements forms its own group, and groups are reduced
independently through the warp/inter-warp phases before being combined at the
end.

**Shared memory sizing** (`getScratchRepShape()`, ~line 122):

```cpp
smemShape[axis] = K * getInterWarpSizeWithUniqueData();
```

Inner tree needs K× more shared memory along the reduction axis to store
partial results from each contiguous group separately.

### Lowering: ReduceOpToLLVM.cpp

**File: `lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp`**

The `ReduceOpConversion` class modifies all six phases of the reduction
lowering when `isInnerTree()` returns true:

#### Phase 1: Within-Thread Reduction (~line 172)

**`reduceWithinThreadsInnerTree()`**: Instead of sequentially accumulating all
registers, this:

1. Groups elements by output position (non-reduced coordinates with axis
   zeroed)
2. Sorts each group by reduction-axis coordinate
3. Splits into contiguous runs along the reduction axis
4. **Tree-reduces within each contiguous group** — pairs adjacent elements,
   then pairs the results, etc.

Each contiguous group produces a separate accumulator. If a thread holds
elements at axis positions {0,1,2,5,6}, it forms two groups: {0,1,2} and
{5,6}, each tree-reduced independently.

#### Phase 2: Within-Warp Reduction (~line 239)

**`warpReduce()`** gains a `countUp` parameter:

- **Default (`countUp=false`)**: Shuffle strides go N/2, N/4, ..., 1
  (standard count-down tree)
- **Inner tree (`countUp=true`)**: Shuffle strides go 1, 2, 4, ..., N/2
  (count-up tree)

Count-up order means the smallest (most local) strides are combined first,
matching the inner-tree convention of reducing neighbors before distant
elements.

#### Phase 3: Store to Shared Memory (~line 376)

**`storeWarpReduceToSharedMemory()`**: For inner tree, writes use offset
`accGroupIdx * sizeInterWarps + warpIdAxis` so each contiguous group occupies
its own SMEM slot, keeping groups separate for the inter-warp phase.

#### Phase 4: Inter-Warp Accumulation (~line 448)

**`accumulatePartialReductions()`**: Passes `countUp=true` to `warpReduce` for
the inter-warp reduction.

#### Phase 5: Load and Final Reduction (~line 510)

**`loadReductionAndPackResult()`**: For K > 1, loads K partial results from
shared memory (one per contiguous group) and tree-reduces them:

```cpp
for (unsigned g = 0; g < K; ++g) {
    // load from readPtr + g * sizeInterWarps * elemSize
}
// pairwise tree-reduce groupVals to single result
```

#### Phase 6: Pack Results (Warp-Synchronous Path) (~line 290)

**`packResults()`**: For inner tree, groups all partial accumulators by
non-axis key and tree-reduces them, analogous to Phase 5 but for the case
where no shared memory is needed (reduction within a single warp).

---

## Why Count-Up vs Count-Down Matters

Consider 8 values: `a b c d e f g h`

**Count-down** (default, stride 4→2→1):
```
Step 1 (stride 4): (a+e) (b+f) (c+g) (d+h)
Step 2 (stride 2): ((a+e)+(c+g)) ((b+f)+(d+h))
Step 3 (stride 1): (((a+e)+(c+g))+((b+f)+(d+h)))
```

**Count-up / inner tree** (stride 1→2→4):
```
Step 1 (stride 1): (a+b) (c+d) (e+f) (g+h)
Step 2 (stride 2): ((a+b)+(c+d)) ((e+f)+(g+h))
Step 3 (stride 4): (((a+b)+(c+d))+((e+f)+(g+h)))
```

The inner tree always combines **neighbors first**, producing a balanced
binary tree over the logical element order. This is independent of how
elements happen to be distributed across threads — the mapping from logical
position to thread is encoded in the layout, but the reduction tree shape is
fixed.

---

## Testing

### Lit Test (LLVM IR Level)

**File: `test/Conversion/reduce_inner_tree_to_llvm.mlir`**

Verifies that inner tree produces count-up shuffle order (strides 2, 4, 8, 16)
in the generated LLVM IR, using a specific linear layout where each register
forms its own contiguous group (K=2).

Compare with the default ordering test in `test/Conversion/reduce_to_llvm.mlir`
which produces count-down shuffle order (strides 16, 8, 4, 2).

### Python Tests (Bitwise Equivalence)

**Reference generation: `python/test/unit/language/generate_reduction_ordering_refs.py`**

Standalone script that generates canonical `.pt` reference tensors using
`num_warps=1` with `INNER_TREE` ordering. Must be run once on a CUDA machine:

```bash
python python/test/unit/language/generate_reduction_ordering_refs.py
```

Produces files in `python/test/unit/language/test_data/`:
- `reduction_ordering_input_{N_ROWS}.pt` — input data (seeded `torch.manual_seed(42)`)
- `reduction_ordering_sum_ref_{N_ROWS}.pt` — expected sum output
- `reduction_ordering_mul_input_{N_ROWS}.pt` — input for multiply (uniform 0.99–1.01)
- `reduction_ordering_mul_ref_{N_ROWS}.pt` — expected multiply output

**Test functions: `python/test/unit/language/test_core.py`**

- `test_reduction_ordering_sum` — `tl.sum` with additive reduction
- `test_reduction_ordering_reduce_mul` — `tl.reduce` with multiplicative combine

Both parametrize over:
- `N_ROWS` ∈ {1, 4, 16, 32} (non-reduction dimension)
- `row_major` ∈ {True, False} (memory layout)

Each test loads the saved input and reference tensors, then runs the kernel
with `num_warps` ∈ {1, 2, 4, 8} and asserts `torch.equal(out, reference)`.

Run:
```bash
pytest python/test/unit/language/test_core.py::test_reduction_ordering_sum \
      python/test/unit/language/test_core.py::test_reduction_ordering_reduce_mul -v
```

---

## Adding a New Reduction Ordering

To add a new ordering strategy (e.g., `OUTER_TREE`):

1. **Python frontend**: Add a new `ReductionOrdering` constant in
   `python/triton/language/core.py` with a unique `name` string.

2. **C++ analysis**: Update `getNumContiguousGroupsOnAxis()` in
   `lib/Analysis/Utility.cpp` if the new strategy changes how shared memory
   is sized.

3. **C++ lowering**: Add the new strategy's logic to each phase in
   `lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp`. The `isInnerTree()`
   pattern can be extended to a switch/enum on the attribute value.

4. **Tests**: Add a lit test in `test/Conversion/` and Python bitwise
   equivalence tests in `test_core.py` with saved reference tensors.

5. **`CompositeReductionOrdering`**: If the strategy is meant to be composed
   with others (e.g., inner tree for within-thread + outer tree for
   across-warp), implement the `CompositeReductionOrdering` path in
   `core.py:reduce()` and extend the C++ side to read a structured attribute
   instead of a single string.
