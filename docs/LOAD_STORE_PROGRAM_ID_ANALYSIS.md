# Load/Store Address Block to Program ID Analysis

This document analyzes how load/store operations' address blocks relate to `program_id` in Triton, and provides a design for implementing an analysis pass.

## 1. Overview

In Triton's SPMD (Single Program Multiple Data) programming model, each program instance is identified by a `program_id` which is used to compute unique memory addresses for load/store operations. Understanding how these addresses are derived from `program_id` is crucial for:

- **Memory coalescing optimization**: Ensuring threads within a warp access contiguous memory
- **Vectorization**: Determining how many elements each thread should handle
- **Alias analysis**: Understanding when different programs access the same or different memory
- **Bounds checking**: Ensuring programs don't access out-of-bounds memory

## 2. Key Triton IR Operations

### 2.1 Program ID Operations

**File**: `/data/users/mren/MetaMain/triton/include/triton/Dialect/Triton/IR/TritonOps.td` (lines 629-647)

```tablegen
def TT_GetProgramIdOp : TT_Op<"get_program_id", [Pure]> {
    let arguments = (ins TT_ProgramDim:$axis);
    let results = (outs I32:$result);
}
```

- Returns a scalar `i32` representing the program ID for a given axis (0, 1, or 2 for 3D grids)
- Pure operation (no side effects)
- Range: `[0, num_programs - 1]` where max is typically `2^31 - 1`

### 2.2 Pointer Arithmetic Operations

**AddPtrOp** (lines 206-219):
```tablegen
def TT_AddPtrOp : TT_Op<"addptr", [Pure, Elementwise, ...]> {
    let arguments = (ins TT_PtrLike:$ptr, TT_IntLike:$offset);
    let results = (outs TT_PtrLike:$result);
}
```

- Element-wise pointer arithmetic: `ptr + offset * sizeof(pointee)`
- Critical for computing per-thread memory addresses

### 2.3 Load/Store Operations

**LoadOp** (lines 239-312):
```tablegen
def TT_LoadOp : TT_Op<"load", [...MemoryEffectsOpInterface...]> {
    let arguments = (
        ins AnyTypeOf<[TT_PtrLike, TT_TensorPtr]>:$ptr,
        Optional<TT_BoolLike>:$mask,
        Optional<TT_Type>:$other,
        ...
    );
}
```

**StoreOp** (lines 314-358):
```tablegen
def TT_StoreOp : TT_Op<"store", [...]> {
    let arguments = (ins
        Arg<AnyTypeOf<[TT_PtrLike, TT_TensorPtr]>, "", [MemWrite<GlobalMemory>]>:$ptr,
        TT_Type:$value,
        Optional<TT_BoolLike>:$mask,
        ...
    );
}
```

## 3. Typical Address Calculation Pattern

A typical Triton program computes addresses as follows:

```
%pid = tt.get_program_id x : i32                    // Get program ID
%block_start = arith.muli %pid, %c128_i32 : i32     // pid * BLOCK_SIZE
%offsets = tt.make_range {start=0, end=128}         // [0, 1, 2, ..., 127]
%block_offsets = arith.addi %block_start, %offsets  // [pid*128, pid*128+1, ..., pid*128+127]
%ptr_base = tt.splat %arg0 : !tt.ptr<f32>           // Broadcast base pointer
%ptrs = tt.addptr %ptr_base, %block_offsets         // Compute final addresses
%vals = tt.load %ptrs : tensor<128xf32>             // Load from computed addresses
```

### 3.1 Data Flow Chain

```
GetProgramIdOp (scalar i32)
    ↓
arith.muli (scale by block size)
    ↓
arith.addi / SplatOp + BroadcastOp (combine with per-thread offsets)
    ↓
AddPtrOp (add to base pointer tensor)
    ↓
LoadOp / StoreOp (memory access)
```

## 4. Existing Analysis Infrastructure

### 4.1 AxisInfo Analysis

**Location**: `/data/users/mren/MetaMain/triton/lib/Analysis/AxisInfo.cpp`

Tracks three properties per tensor dimension:
- **Contiguity**: Length of shortest sequence of contiguous integers
- **Divisibility**: Largest power-of-2 dividing first element of each contiguous sequence
- **Constancy**: Length of shortest repeating sequence

Example for `AddPtrOp` handling (lines 290-308):
```cpp
if constexpr (std::is_same_v<OpTy, triton::AddPtrOp>) {
    auto elemSize = triton::getPointeeBitWidth(op.getPtr().getType()) / 8;
    rhsDivisibility = multiplyDivisor(rhs.getDivisibility(dim), elemSize);
}
return gcd(lhs.getDivisibility(dim), rhsDivisibility);
```

### 4.2 RangeAnalysis (AMD)

**Location**: `/data/users/mren/MetaMain/triton/third_party/amd/lib/Analysis/RangeAnalysis.cpp`

Extends MLIR's `IntegerRangeAnalysis` for Triton:
- Tracks value ranges (min, max) for integer values
- Special handling for `GetProgramIdOp`: range is `[0, 2^31 - 1]`
- Supports "abstract interpretation" through loops
- Uses assumptions (e.g., from `tl.assume`) for tighter bounds

```cpp
// Lines 612-616: GetProgramIdOp range inference
if (llvm::isa<GetProgramIdOp, MakeRangeOp, HistogramOp, GetNumProgramsOp>(op)) {
    llvm::TypeSwitch<Operation *>(op)
        .Case<GetProgramIdOp>([&](auto getPIDOp) {
            inferResultRangesPID(getPIDOp, kDefaultMaxPrograms - 1, joinCallback);
        })
```

### 4.3 Coalescing Analysis

**Location**: `/data/users/mren/MetaMain/triton/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp`

Uses AxisInfo to determine optimal memory access patterns:
```cpp
// Get memory access contiguity from axis info
auto contiguity = axisInfoAnalysis.getAxisInfo(ptr)->getContiguity();
SmallVector<unsigned> order = getOrderFromContiguity(contiguity);
```

## 5. Proposed Analysis Pass Design

### 5.1 Analysis Goal

Track the **affine relationship** between each load/store address and `program_id`:

```
address[i] = base_ptr + (pid * stride + offset[i]) * elem_size
```

Where:
- `base_ptr`: The base pointer argument
- `pid`: The program ID(s) used
- `stride`: The multiplier for program_id (typically block size)
- `offset[i]`: Per-element offset within the block
- `elem_size`: Size of element being accessed

### 5.2 Lattice State Definition

```cpp
// Represents the relationship: addr = base + (pid * stride + offset) * elem_size
class ProgramIdAddressInfo {
public:
    struct AddressExpr {
        Value basePtr;                    // Base pointer value
        SmallVector<int> pidAxes;         // Which program_id axes are used (0, 1, 2)
        SmallVector<int64_t> pidStrides;  // Stride for each pid axis
        int64_t constantOffset;           // Constant offset component
        bool hasNonConstantOffset;        // true if offset depends on other values
        int64_t elemSize;                 // Size of accessed element
    };

    std::optional<AddressExpr> expr;      // None if relationship is unknown

    static ProgramIdAddressInfo join(const ProgramIdAddressInfo& lhs,
                                     const ProgramIdAddressInfo& rhs);
    static ProgramIdAddressInfo getPessimisticValueState(Value v);
};
```

### 5.3 Pass Implementation Structure

```cpp
class ProgramIdToAddressAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<
          dataflow::Lattice<ProgramIdAddressInfo>> {
private:
    // Visitors for different operations
    class GetProgramIdVisitor;
    class ArithMulIVisitor;
    class ArithAddIVisitor;
    class SplatOpVisitor;
    class AddPtrOpVisitor;
    class LoadOpVisitor;
    class StoreOpVisitor;

public:
    void setToEntryState(dataflow::Lattice<ProgramIdAddressInfo> *lattice) override;

    LogicalResult visitOperation(
        Operation *op,
        ArrayRef<const dataflow::Lattice<ProgramIdAddressInfo> *> operands,
        ArrayRef<dataflow::Lattice<ProgramIdAddressInfo> *> results) override;
};
```

### 5.4 Key Transfer Functions

#### GetProgramIdOp:
```cpp
AxisInfo getAxisInfo(GetProgramIdOp op, ...) {
    return ProgramIdAddressInfo{
        .basePtr = nullptr,
        .pidAxes = {op.getAxisAsInt()},
        .pidStrides = {1},
        .constantOffset = 0,
        .hasNonConstantOffset = false,
        .elemSize = 1
    };
}
```

#### arith.muli (pid * constant):
```cpp
// If one operand is pid info and other is constant
if (lhs.isPidBased() && rhs.isConstant()) {
    return ProgramIdAddressInfo{
        .pidAxes = lhs.pidAxes,
        .pidStrides = lhs.pidStrides * rhs.getConstantValue(),
        ...
    };
}
```

#### tt.addptr:
```cpp
// Combines pointer base with offset
ProgramIdAddressInfo getAxisInfo(AddPtrOp op, ...) {
    auto ptrInfo = operands[0]->getValue();
    auto offsetInfo = operands[1]->getValue();

    return ProgramIdAddressInfo{
        .basePtr = /* extract from ptrInfo or op.getPtr() */,
        .pidAxes = /* merge from both */,
        .pidStrides = offsetInfo.pidStrides * elemSize,
        .constantOffset = offsetInfo.constantOffset * elemSize,
        ...
    };
}
```

### 5.5 Output: Load/Store Classification

The analysis produces for each load/store:

```cpp
struct LoadStoreAddressPattern {
    enum class Pattern {
        PID_INDEPENDENT,      // Address doesn't depend on program_id
        PID_AFFINE,           // addr = base + pid * stride + offset
        PID_MULTI_AXIS,       // Depends on multiple pid axes
        PID_NONLINEAR,        // Non-affine dependence on pid
        UNKNOWN               // Cannot determine relationship
    };

    Pattern pattern;
    Value basePtr;
    SmallVector<std::pair<int, int64_t>> pidAxisStrides;  // [(axis, stride), ...]
    std::optional<int64_t> blockSize;   // Size of contiguous block per program
    bool isCoalesced;                    // Whether access pattern is coalesced
};
```

## 6. Implementation Steps

### Step 1: Define the Lattice
Create `ProgramIdAddressInfo.h` in `/data/users/mren/MetaMain/triton/include/triton/Analysis/`

### Step 2: Implement Transfer Functions
Create `ProgramIdAddressInfo.cpp` in `/data/users/mren/MetaMain/triton/lib/Analysis/`

Implement visitors for:
- `GetProgramIdOp`: Initialize pid info
- `arith.muli`, `arith.addi`, `arith.subi`: Propagate through arithmetic
- `tt.splat`, `tt.broadcast`, `tt.expand_dims`: Handle tensor shape changes
- `tt.addptr`: Combine pointer and offset info
- `tt.load`, `tt.store`: Record final address pattern

### Step 3: Register the Pass
Add pass definition to `/data/users/mren/MetaMain/triton/include/triton/Analysis/Passes.td`

### Step 4: Add Tests
Create test file in `/data/users/mren/MetaMain/triton/test/Analysis/test-pid-address.mlir`

### Step 5: Integration
Optionally expose results for use by other passes (e.g., coalescing, vectorization)

## 7. Example Analysis Results

### Input IR:
```mlir
tt.func @vector_add(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c128 = arith.constant 128 : i32
    %pid = tt.get_program_id x : i32
    %block_start = arith.muli %pid, %c128 : i32
    %offsets = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %indices = arith.addi %block_start, %offsets : tensor<128xi32>

    %ptr_a = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %ptrs_a = tt.addptr %ptr_a, %indices : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %a = tt.load %ptrs_a : tensor<128x!tt.ptr<f32>>
    ...
}
```

### Analysis Output:
```
%pid:       PID_INFO(axis=0, stride=1, offset=0)
%block_start: PID_INFO(axis=0, stride=128, offset=0)
%indices:   PID_INFO(axis=0, stride=128, offset=[0,127], elem_range)
%ptrs_a:    ADDR_INFO(base=%arg0, axis=0, stride=512, offset=[0,508], coalesced=true)
%a (load):  PATTERN(PID_AFFINE, block_size=128, stride=512, coalesced=true)
```

## 8. Potential Applications

1. **Memory Coalescing Verification**: Verify that memory accesses are coalesced
2. **Vectorization Hints**: Determine optimal vector width based on address patterns
3. **Out-of-Bounds Detection**: Detect potential OOB accesses when bounds are known
4. **Alias Analysis**: Determine when different programs access disjoint memory
5. **Prefetching**: Guide prefetch insertion based on access patterns

## 9. References

- AxisInfo Analysis: `/data/users/mren/MetaMain/triton/lib/Analysis/AxisInfo.cpp`
- RangeAnalysis: `/data/users/mren/MetaMain/triton/third_party/amd/lib/Analysis/RangeAnalysis.cpp`
- Coalesce Pass: `/data/users/mren/MetaMain/triton/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp`
- MLIR DataFlow Framework: `mlir/Analysis/DataFlow/SparseAnalysis.h`
- Triton IR Operations: `/data/users/mren/MetaMain/triton/include/triton/Dialect/Triton/IR/TritonOps.td`
