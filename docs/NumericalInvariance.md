# TTGIR Numerical Invariance Mapping

## Overview

This document describes a design for mapping TTGIR (Triton GPU IR) modules into a **numerical invariance** - a canonical representation that captures all numerically-significant properties while ignoring superficial differences.

**Primary Use Case**: Regression prevention for numerical mismatches. If two TTGIRs have the same numerical invariance, they should produce bitwise-identical outputs for the same inputs.

## Core Concept

```
TTGIR Module A ──┐
                 ├──> Numerical Invariance ──> Fingerprint (hash)
TTGIR Module B ──┘
```

Two TTGIR modules with **different syntax** but **same numerical behavior** should map to the **same invariance/fingerprint**.

---

## What to Capture vs. Ignore

### Properties to CAPTURE (Numerically Significant)

| Property | Why It Matters | Example |
|----------|----------------|---------|
| **Data types** | Fundamental precision | `fp32` vs `fp16` vs `bf16` |
| **Operation semantics** | Defines computation | `tt.dot`, `tt.reduce`, `arith.addf` |
| **Computation DAG structure** | Data flow determines result | Order of ops, producer-consumer relationships |
| **Layout encodings** | Thread-to-element mapping affects reduction order | `#blocked`, `#mma`, `#shared` |
| **Shared memory swizzling** | Memory access patterns, bank conflicts | `vec=8, perPhase=1, maxPhase=8` |
| **MMA/MFMA version** | Different hardware units, different numerical properties | MMA v2 vs v3 |
| **Reduction semantics** | Floating-point associativity | Reduction axis, order |
| **Instruction shape** | Tensor core behavior | `instrShape = [16, 8, 16]` |

### Properties to IGNORE (Cosmetic/Superficial)

| Property | Why Ignore | Example |
|----------|------------|---------|
| **SSA value names** | Arbitrary naming, same data flow | `%0` vs `%result` |
| **Instruction ordering** (independent ops) | No data dependency = no numerical effect | Swapping independent loads |
| **Block argument names** | Arbitrary naming | `^bb0(%arg0:...)` |
| **Debug info / locations** | No runtime effect | `loc(#loc123)` |
| **Comments / metadata** | No runtime effect | |
| **Module/function names** | Naming only | `@kernel_v1` vs `@kernel_v2` |

---

## Invariance Structure

```cpp
struct NumericalInvariance {
    // Level 1: Computation Graph Structure
    // - DAG of operations with edges representing data flow
    // - Ignores value names, uses structural hashing
    size_t computation_dag_hash;
    
    // Level 2: Type Signature
    // - Function input/output types
    // - All intermediate tensor types
    std::string dtype_signature;
    
    // Level 3: Layout Information  
    // - All tensor layout encodings (blocked, mma, shared, etc.)
    // - Shared memory configurations
    size_t layout_hash;
    
    // Level 4: Hardware-Specific Configuration
    // - MMA version and instruction shapes
    // - Swizzling parameters
    size_t hw_config_hash;
    
    // Combined fingerprint
    size_t fingerprint() const {
        return llvm::hash_combine(
            computation_dag_hash,
            dtype_signature,
            layout_hash,
            hw_config_hash
        );
    }
};
```

---

## Algorithm Design

### Phase 1: Canonicalize the Computation DAG

For each operation, create a **canonical operation descriptor**:

```cpp
struct CanonicalOpDescriptor {
    StringRef opName;           // e.g., "tt.dot", "arith.addf"
    SmallVector<Type> operandTypes;
    SmallVector<Type> resultTypes;
    DenseMap<StringRef, Attribute> attrs;  // Sorted by name
    SmallVector<size_t> operandHashes;     // Hash of producer ops
};
```

**Key insight**: Instead of using SSA value names, we use **structural hashing** - each operation's hash is computed from its opcode, types, attributes, and the hashes of its operand producers.

### Phase 2: Extract Layout Information

For each tensor type in the IR:

```cpp
struct LayoutDescriptor {
    ArrayRef<int64_t> shape;
    Type elementType;
    Attribute encoding;  // The full layout encoding
};

size_t hashLayout(const LayoutDescriptor& ld) {
    // Use MLIR's built-in attribute hashing
    return llvm::hash_combine(
        llvm::hash_combine_range(ld.shape.begin(), ld.shape.end()),
        mlir::hash_value(ld.elementType),
        mlir::hash_value(ld.encoding)
    );
}
```

### Phase 3: Extract Hardware Configuration

```cpp
struct HWConfig {
    // MMA Configuration
    std::optional<int> mmaVersionMajor;
    std::optional<int> mmaVersionMinor;
    std::optional<SmallVector<int>> instrShape;
    
    // MFMA Configuration  
    std::optional<int> mfmaVersion;
    std::optional<Type> mfmaElementType;
    
    // Shared Memory Configuration
    struct SharedMemConfig {
        int vec;
        int perPhase;
        int maxPhase;
        SmallVector<unsigned> order;
    };
    SmallVector<SharedMemConfig> sharedConfigs;
};
```

### Phase 4: Compute Final Fingerprint

```cpp
size_t computeNumericalFingerprint(ModuleOp module) {
    NumericalInvariance inv;
    
    // Walk all operations in topological order
    module.walk([&](Operation* op) {
        // Update computation_dag_hash
        inv.computation_dag_hash = llvm::hash_combine(
            inv.computation_dag_hash,
            computeOpHash(op)
        );
        
        // Extract layout info from tensor types
        for (Type t : op->getResultTypes()) {
            if (auto tensorTy = dyn_cast<RankedTensorType>(t)) {
                inv.layout_hash = llvm::hash_combine(
                    inv.layout_hash,
                    hashLayout(tensorTy)
                );
            }
        }
        
        // Extract HW config from specific ops/attrs
        updateHWConfig(op, &inv.hw_config_hash);
    });
    
    return inv.fingerprint();
}
```

---

## Example: Two TTGIRs with Same Invariance

### TTGIR A:
```mlir
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], ...}>
module {
  tt.func @matmul(%arg0: tensor<128x64xf16, #blocked>) {
    %0 = tt.dot %arg0, %arg1 : tensor<...> -> tensor<128x128xf32, #blocked>
    tt.return
  }
}
```

### TTGIR B:
```mlir
#layout0 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], ...}>
module {
  tt.func @matmul_v2(%input: tensor<128x64xf16, #layout0>) {
    %result = tt.dot %input, %weights : tensor<...> -> tensor<128x128xf32, #layout0>
    tt.return
  }
}
```

**Same Invariance** because:
- Same operation types (`tt.dot`)
- Same tensor shapes and element types
- Same layout encoding parameters
- SSA names (`%0` vs `%result`) and function names (`@matmul` vs `@matmul_v2`) are ignored

---

## Example: Two TTGIRs with Different Invariance

### TTGIR A (MMA v2):
```mlir
#mma = #ttg.nvidia_mma<{versionMajor = 2, instrShape = [16, 8, 16], ...}>
```

### TTGIR B (MMA v3):
```mlir
#mma = #ttg.nvidia_mma<{versionMajor = 3, instrShape = [16, 16, 16], ...}>
```

**Different Invariance** because:
- Different MMA versions (v2 vs v3)
- Different instruction shapes
- These can produce numerically different results due to different tensor core implementations

---

## Implementation Entry Points

1. **Header**: `/include/triton/Dialect/TritonGPU/Transforms/NumericalInvariance.h`
2. **Implementation**: `/lib/Dialect/TritonGPU/Transforms/NumericalInvariance.cpp`
3. **Pass Registration**: Add a pass that computes and optionally prints/stores the invariance

### Integration with Existing Infrastructure

- Use `llvm::hash_combine()` from `/include/triton/Dialect/TritonGPU/IR/Dialect.h`
- Use `getLayoutStr()` from `Dialect.cpp` for debugging/printing
- Build on `CacheKey` pattern for (shape, attribute) hashing

---

## Usage for Regression Prevention

### Testing Workflow

1. **Baseline**: Compute numerical invariance for a known-good TTGIR
2. **After Change**: Compute numerical invariance for the modified TTGIR
3. **Compare**:
   - Same invariance → **No numerical regression expected**
   - Different invariance → **Potential numerical difference, investigate**

### CI Integration

```python
def test_no_numerical_regression():
    baseline_ttgir = load_baseline()
    current_ttgir = compile_kernel()
    
    baseline_fp = compute_numerical_fingerprint(baseline_ttgir)
    current_fp = compute_numerical_fingerprint(current_ttgir)
    
    if baseline_fp != current_fp:
        # Detailed diff showing what changed
        diff = compute_invariance_diff(baseline_ttgir, current_ttgir)
        if is_expected_change(diff):
            update_baseline()
        else:
            fail("Unexpected numerical change detected")
```

---

## Future Extensions

1. **Hierarchical Invariance**: Compute invariance at different granularities (op, block, function, module)
2. **Semantic Equivalence**: Detect more complex equivalences (e.g., `a * 2` ≡ `a + a` for integers)
3. **Precision Analysis**: Track precision loss through the computation graph
4. **Golden Reference Storage**: Store invariances alongside golden outputs for regression testing
