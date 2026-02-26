//===----------------------------------------------------------------------===//
// TLX Print TTGIR to TLX Pass
//===----------------------------------------------------------------------===//
//
// This pass converts Triton GPU IR (TTGIR) to a simplified TLX-style
// representation for debugging and understanding the correspondence between
// high-level TLX Python API and low-level GPU IR.
//
// Key Features:
// - Converts TTGIR operations to their TLX equivalents (e.g., ttng.wait_barrier
//   -> tlx.barrier_wait)
// - Removes layouts, types, and attributes for readability
// - Uses Python-like syntax for control flow:
//   * scf.for -> for var in range(start, end, step):
//   * scf.if -> if condition: / else:
//   * ttg.warp_specialize -> with tlx.async_tasks(): / with tlx.async_task():
// - Smart local_alloc handling:
//   * Barrier allocations -> tlx.alloc_barriers(count)
//   * Buffer allocations -> tlx.local_alloc((shape), dtype, count)
// - Variable name simplification:
//   * Removes % prefix from SSA names
//   * Prefixes numeric-only names with "var_" (e.g., %0 -> var_0)
// - Argument substitution:
//   * warp_specialize partition args -> original operands
//   * scf.for outputs -> corresponding iter_args
// - Implicit control flow:
//   * scf.yield inside if -> assignment to if's output variable
//   * scf.yield inside for -> skipped (iter_args updated via block args)
//   * ttg.warp_yield, ttg.warp_return -> skipped (implicit in with blocks)
//
// Example output:
//   func _attn_fwd_persist(arg0, arg1, arg2, arg3) {
//     c0_i32 = 0
//     c1_i32 = 1
//     var_0 = tlx.alloc_barriers(1)
//     var_92 = tlx.local_alloc((128, 128), bf16, 3)
//     with tlx.async_tasks():
//       with tlx.async_task("default"):
//         var_97 = get_program_id()
//         if var_103:
//           var_108 = add(var_101, c1_i32)
//           var_104 = var_108
//         else:
//           var_104 = var_101
//         arg9 = var_97
//         for arg8 in range(c0_i32, var_104, c1_i32):
//           tlx.barrier_wait(var_120, var_122, true)
//           tlx.tc_gen5_mma(...)
//       with tlx.async_task():
//         ... partition code ...
//   }
//
// Usage:
//   triton-opt --tlx-print-ttgir-to-tlx input.mlir
// Or via environment variable:
//   TRITON_DUMP_TTGIR_TO_TLX=1 python your_kernel.py
//
//===----------------------------------------------------------------------===//

#include "IR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <vector>

#define DEBUG_TYPE "tlx-print-ttgir-to-tlx"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace tlx {

#define GEN_PASS_DEF_TLXPRINTTTGIRTOTLX
#include "tlx/dialect/include/Transforms/Passes.h.inc"

namespace {

struct TTGIRToTLXMapping {
  StringRef ttgirOpName;
  StringRef tlxOpName;
  StringRef description;
};

static const TTGIRToTLXMapping opMappings[] = {
    // Barrier operations - init_barrier is handled specially
    {"ttng.barrier_expect", "tlx.barrier_expect_bytes",
     "Set expected bytes for barrier transaction tracking"},
    {"ttng.wait_barrier", "tlx.barrier_wait",
     "Wait for barrier phase completion"},
    {"ttng.arrive_barrier", "tlx.barrier_arrive", "Signal arrival at barrier"},
    {"ttng.named_barrier_wait", "tlx.named_barrier_wait",
     "Wait on named hardware barrier"},
    {"ttng.named_barrier_arrive", "tlx.named_barrier_arrive",
     "Arrive at named hardware barrier"},

    // Memory allocation operations - local_alloc is handled specially
    {"ttng.tmem_alloc", "tlx.tmem_alloc",
     "Allocate tensor memory buffer (Blackwell)"},

    // Memory load/store operations
    {"ttg.local_load", "tlx.local_load",
     "Load from shared memory to registers"},
    {"ttng.tmem_load", "tlx.tmem_load",
     "Load from tensor memory to registers (Blackwell)"},
    {"ttg.local_store", "tlx.local_store", "Store registers to shared memory"},
    {"ttng.tmem_store", "tlx.tmem_store",
     "Store registers to tensor memory (Blackwell)"},
    {"ttng.tmem_copy", "tlx.tmem_copy",
     "Copy from shared memory to tensor memory (Blackwell)"},

    // Memory descriptor operations
    {"ttg.memdesc_subview", "tlx.local_view", "Get subview of a buffer"},
    {"ttg.memdesc_trans", "tlx.local_trans", "Transpose buffer dimensions"},
    {"ttg.memdesc_reinterpret", "tlx.local_reinterpret",
     "Reinterpret buffer dtype/shape"},
    {"ttng.tmem_subslice", "tlx.subslice", "TMEM subslice (Blackwell)"},
    {"ttg.memdesc_index", "tlx.memdesc_index", "Index into memdesc"},

    // Async copy operations (cp.async)
    {"ttg.async_load", "tlx.async_load",
     "Async load from global to shared memory"},
    {"ttg.async_commit_group", "tlx.async_load_commit_group",
     "Commit async load group"},
    {"ttg.async_wait", "tlx.async_load_wait_group",
     "Wait for async load completion"},

    // TMA operations
    {"ttng.async_tma_copy_global_to_local", "tlx.async_descriptor_load",
     "TMA load from global to shared memory"},
    {"ttng.async_tma_copy_local_to_global", "tlx.async_descriptor_store",
     "TMA store from shared to global memory"},
    {"ttng.tma_store_wait", "tlx.async_descriptor_store_wait",
     "Wait for TMA stores to complete"},
    {"ttng.async_tma_store_wait", "tlx.async_descriptor_store_wait",
     "Wait for TMA stores to complete"},
    {"tt.make_tensor_descriptor", "tlx.make_tensor_descriptor",
     "Create TMA descriptor on device"},
    {"tt.reinterpret_tensor_descriptor", "tlx.reinterpret_tensor_descriptor",
     "Reinterpret TMA descriptor with new shape"},

    // MMA operations
    {"ttng.warp_group_dot", "tlx.warp_group_dot",
     "Warp-group MMA (Hopper wgmma.mma_async)"},
    {"ttng.warp_group_dot_wait", "tlx.warp_group_dot_wait",
     "Wait for async dot completion"},
    {"ttng.tc_gen5_mma", "tlx.tc_gen5_mma",
     "Tensor Core Gen5 MMA (Blackwell tcgen05.mma)"},
    {"ttng.tc_gen5_mma_scaled", "tlx.tc_gen5_mma_scaled",
     "Scaled FP8 MMA (Blackwell)"},
    {"ttng.tc_gen5_commit", "tlx.tc_gen5_commit",
     "Commit tcgen05 operations to barrier (Blackwell)"},

    // Fence operations
    {"ttng.fence_async_shared", "tlx.fence_async_shared",
     "Memory fence for shared memory ordering"},

    // Remote memory operations
    {"ttng.map_to_remote_buffer", "tlx.remote_view",
     "Map buffer to remote CTA in cluster"},
    {"ttng.remote_store", "tlx.remote_shmem_store",
     "Store to remote CTA's shared memory"},
    {"ttng.async_remote_store", "tlx.async_remote_shmem_store",
     "Async store to remote CTA's shared memory"},

    // Warp specialization
    {"ttg.warp_specialize", "tlx.warp_specialize",
     "Warp specialization region"},
    {"ttg.warp_return", "tlx.warp_return",
     "Return from warp specialization region"},

    // Control flow
    {"scf.for", "for", "For loop"},
    {"scf.if", "if", "If statement"},
    {"scf.yield", "yield", "Yield values"},
    {"scf.while", "while", "While loop"},

    // Arith operations
    {"arith.constant", "const", "Constant value"},
    {"arith.addi", "add", "Integer addition"},
    {"arith.subi", "sub", "Integer subtraction"},
    {"arith.muli", "mul", "Integer multiplication"},
    {"arith.divsi", "div", "Signed integer division"},
    {"arith.remsi", "rem", "Signed integer remainder"},
    {"arith.addf", "addf", "Float addition"},
    {"arith.subf", "subf", "Float subtraction"},
    {"arith.mulf", "mulf", "Float multiplication"},
    {"arith.divf", "divf", "Float division"},
    {"arith.cmpf", "cmpf", "Float comparison"},
    {"arith.cmpi", "cmpi", "Integer comparison"},
    {"arith.select", "select", "Select operation"},
    {"arith.extui", "extui", "Extend unsigned integer"},
    {"arith.extsi", "extsi", "Extend signed integer"},
    {"arith.trunci", "trunci", "Truncate integer"},
    {"arith.xori", "xor", "Integer XOR"},
    {"arith.andi", "and", "Integer AND"},
    {"arith.ori", "or", "Integer OR"},
    {"arith.maxf", "maxf", "Float max"},
    {"arith.minf", "minf", "Float min"},
    {"arith.negf", "negf", "Float negation"},
    {"arith.sitofp", "sitofp", "Signed int to float"},
    {"arith.fptosi", "fptosi", "Float to signed int"},
    {"arith.uitofp", "uitofp", "Unsigned int to float"},
    {"arith.fptoui", "fptoui", "Float to unsigned int"},
    {"arith.truncf", "truncf", "Truncate float"},
    {"arith.extf", "extf", "Extend float"},
    {"arith.bitcast", "bitcast", "Bitcast"},

    // Triton operations
    {"tt.splat", "splat", "Splat scalar to tensor"},
    {"tt.broadcast", "broadcast", "Broadcast tensor"},
    {"tt.expand_dims", "expand_dims", "Expand dimensions"},
    {"tt.reduce", "reduce", "Reduce operation"},
    {"tt.dot", "dot", "Matrix multiply"},
    {"tt.load", "load", "Load from global memory"},
    {"tt.store", "store", "Store to global memory"},
    {"tt.addptr", "addptr", "Add to pointer"},
    {"tt.make_range", "make_range", "Make range"},
    {"tt.trans", "trans", "Transpose"},
    {"tt.reshape", "reshape", "Reshape tensor"},
    {"tt.cat", "cat", "Concatenate"},
    {"tt.join", "join", "Join tensors"},
    {"tt.split", "split", "Split tensor"},
    {"tt.get_program_id", "get_program_id", "Get program ID"},
    {"tt.get_num_programs", "get_num_programs", "Get number of programs"},
    {"tt.return", "return", "Return from function"},

    // GPU operations
    {"gpu.barrier", "gpu.barrier", "GPU barrier"},
};

// Build a lookup map for fast operation name lookup
llvm::StringMap<StringRef> buildOpNameMap() {
  llvm::StringMap<StringRef> map;
  for (const auto &mapping : opMappings) {
    map[mapping.ttgirOpName] = mapping.tlxOpName;
  }
  return map;
}

// Get simplified name for a value (just the SSA name)
// If argSubstitutionMap is provided, substitute block args with their mapped
// values
std::string
getValueName(Value v,
             const DenseMap<Value, Value> *argSubstitutionMap = nullptr) {
  // Check if this value should be substituted
  if (argSubstitutionMap) {
    auto it = argSubstitutionMap->find(v);
    if (it != argSubstitutionMap->end()) {
      // Recursively get the name of the substituted value (without
      // substitution)
      return getValueName(it->second, nullptr);
    }
  }

  std::string name;
  llvm::raw_string_ostream os(name);
  v.printAsOperand(os, OpPrintingFlags());
  os.flush();
  // Remove type info if present (after ':')
  size_t colonPos = name.find(':');
  if (colonPos != std::string::npos) {
    name = name.substr(0, colonPos);
  }
  // Trim whitespace
  while (!name.empty() && name.back() == ' ')
    name.pop_back();
  // Remove % prefix
  if (!name.empty() && name[0] == '%') {
    name = name.substr(1);
  }
  // If name is all digits, prefix with "var_"
  if (!name.empty() && std::all_of(name.begin(), name.end(),
                                   [](char c) { return std::isdigit(c); })) {
    name = "var_" + name;
  }
  return name;
}

// Print a constant value
void printConstantValue(Attribute attr, llvm::raw_ostream &os) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    // Special handling for i1 (boolean) type
    if (intAttr.getType().isInteger(1)) {
      os << (intAttr.getValue().getBoolValue() ? "true" : "false");
    } else {
      os << intAttr.getValue();
    }
  } else if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
    SmallString<16> str;
    floatAttr.getValue().toString(str);
    os << str;
  } else if (auto denseAttr = dyn_cast<DenseElementsAttr>(attr)) {
    // For dense tensors, print a summary
    if (denseAttr.isSplat()) {
      os << "splat(";
      printConstantValue(denseAttr.getSplatValue<Attribute>(), os);
      os << ")";
    } else {
      os << "dense<...>";
    }
  } else if (auto boolAttr = dyn_cast<BoolAttr>(attr)) {
    os << (boolAttr.getValue() ? "true" : "false");
  } else {
    // Fallback for other types
    os << "const";
  }
}

// Get element type name as a simple string
std::string getElementTypeName(Type type) {
  if (type.isF32())
    return "f32";
  if (type.isF16())
    return "f16";
  if (type.isBF16())
    return "bf16";
  if (type.isF64())
    return "f64";
  if (type.isInteger(1))
    return "i1";
  if (type.isInteger(8))
    return "i8";
  if (type.isInteger(16))
    return "i16";
  if (type.isInteger(32))
    return "i32";
  if (type.isInteger(64))
    return "i64";
  // Fallback
  std::string str;
  llvm::raw_string_ostream os(str);
  type.print(os);
  return str;
}

// Struct to hold analysis info about local_alloc operations
struct LocalAllocInfo {
  bool isBarrierAlloc = false;
  int barrierCount = 0;
  // For regular allocs: shape (excluding first dim which is count),
  // element type, count
  SmallVector<int64_t> shape;
  Type elementType;
  int64_t bufferCount = 1;
};

// Analyze if a local_alloc is used for barriers
// Returns true if it's a barrier alloc, and counts the number of barriers
LocalAllocInfo analyzeLocalAlloc(Operation *localAllocOp) {
  LocalAllocInfo info;

  if (localAllocOp->getNumResults() == 0)
    return info;

  Value allocResult = localAllocOp->getResult(0);

  // Get the memdesc type to extract shape info
  if (auto memDescType = dyn_cast<ttg::MemDescType>(allocResult.getType())) {
    ArrayRef<int64_t> shape = memDescType.getShape();
    info.elementType = memDescType.getElementType();

    // Check if any use chain leads to init_barrier
    // Pattern: local_alloc -> memdesc_index -> init_barrier
    bool foundInitBarrier = false;
    int initBarrierCount = 0;

    for (Operation *user : allocResult.getUsers()) {
      if (user->getName().getStringRef() == "ttg.memdesc_index") {
        // Check if memdesc_index result is used by init_barrier
        for (Value result : user->getResults()) {
          for (Operation *indexUser : result.getUsers()) {
            if (indexUser->getName().getStringRef() == "ttng.init_barrier") {
              foundInitBarrier = true;
              initBarrierCount++;
            }
          }
        }
      }
    }

    if (foundInitBarrier && info.elementType.isInteger(64)) {
      // This is a barrier allocation
      info.isBarrierAlloc = true;
      // Barrier count is from the first dimension of the shape
      // For !ttg.memdesc<3x1xi64>, we have 3 barriers
      if (!shape.empty()) {
        info.barrierCount = shape[0];
        // If shape is like <1x1xi64>, it's 1 barrier
        // If shape is like <3x1xi64>, it's 3 barriers
      }
    } else {
      // Regular buffer allocation
      info.isBarrierAlloc = false;
      // Shape format: first dim is buffer count, rest is actual shape
      // E.g., <1x128x128xbf16> -> count=1, shape=(128,128)
      // E.g., <3x128x64xf32> -> count=3, shape=(128,64)
      if (shape.size() >= 2) {
        info.bufferCount = shape[0];
        for (size_t i = 1; i < shape.size(); ++i) {
          info.shape.push_back(shape[i]);
        }
      } else if (shape.size() == 1) {
        info.bufferCount = 1;
        info.shape.push_back(shape[0]);
      }
    }
  }

  return info;
}

// Check if an operation should be skipped because it's folded into
// a barrier alloc
bool shouldSkipOp(Operation *op,
                  const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
                  llvm::DenseSet<Operation *> &skippedOps) {
  StringRef opName = op->getName().getStringRef();

  // Skip init_barrier - it's folded into alloc_barriers
  if (opName == "ttng.init_barrier") {
    return true;
  }

  // Skip warp_return - it's implicit in the with block structure
  if (opName == "ttg.warp_return") {
    return true;
  }

  // Skip warp_yield - it's implicit in the with block structure
  if (opName == "ttg.warp_yield") {
    return true;
  }

  // Skip warp_specialize.partitions - it's not meaningful in TLX format
  if (opName == "ttg.warp_specialize.partitions") {
    return true;
  }

  // Skip memdesc_index that are only used by init_barrier for barrier allocs
  if (opName == "ttg.memdesc_index") {
    // Check if operand comes from a barrier alloc
    if (op->getNumOperands() > 0) {
      Value src = op->getOperand(0);
      if (Operation *srcOp = src.getDefiningOp()) {
        if (srcOp->getName().getStringRef() == "ttg.local_alloc") {
          auto it = allocInfoMap.find(srcOp);
          if (it != allocInfoMap.end() && it->second.isBarrierAlloc) {
            // Check if all uses of this memdesc_index are init_barrier
            bool allUsesAreInitBarrier = true;
            for (Value result : op->getResults()) {
              for (Operation *user : result.getUsers()) {
                if (user->getName().getStringRef() != "ttng.init_barrier") {
                  allUsesAreInitBarrier = false;
                  break;
                }
              }
              if (!allUsesAreInitBarrier)
                break;
            }
            if (allUsesAreInitBarrier) {
              return true;
            }
          }
        }
      }
    }
  }

  return skippedOps.count(op) > 0;
}

// Forward declarations
void printRegion(
    Region &region, llvm::raw_ostream &os,
    const llvm::StringMap<StringRef> &opNameMap,
    const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
    llvm::DenseSet<Operation *> &skippedOps, unsigned indent,
    DenseMap<Value, Value> *argSubstitutionMap = nullptr,
    ArrayRef<Value> yieldTargets = {},
    const llvm::DenseMap<Operation *, std::string> *opToHelper = nullptr,
    const llvm::DenseMap<Operation *, std::vector<Value>> *opToInputs = nullptr,
    const llvm::DenseMap<Operation *, std::vector<Operation *>> *opToCoAnchors =
        nullptr);

// Print scf.for in Python range syntax
void printForOp(
    Operation *op, llvm::raw_ostream &os,
    const llvm::StringMap<StringRef> &opNameMap,
    const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
    llvm::DenseSet<Operation *> &skippedOps, unsigned indent,
    DenseMap<Value, Value> *argSubstitutionMap = nullptr,
    const llvm::DenseMap<Operation *, std::string> *opToHelper = nullptr,
    const llvm::DenseMap<Operation *, std::vector<Value>> *opToInputs = nullptr,
    const llvm::DenseMap<Operation *, std::vector<Operation *>> *opToCoAnchors =
        nullptr);

// Print scf.if with yield-to-assignment conversion
void printIfOp(
    Operation *op, llvm::raw_ostream &os,
    const llvm::StringMap<StringRef> &opNameMap,
    const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
    llvm::DenseSet<Operation *> &skippedOps, unsigned indent,
    DenseMap<Value, Value> *argSubstitutionMap = nullptr,
    const llvm::DenseMap<Operation *, std::string> *opToHelper = nullptr,
    const llvm::DenseMap<Operation *, std::vector<Value>> *opToInputs = nullptr,
    const llvm::DenseMap<Operation *, std::vector<Operation *>> *opToCoAnchors =
        nullptr);

// Print scf.for in Python range syntax
void printForOp(
    Operation *op, llvm::raw_ostream &os,
    const llvm::StringMap<StringRef> &opNameMap,
    const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
    llvm::DenseSet<Operation *> &skippedOps, unsigned indent,
    DenseMap<Value, Value> *argSubstitutionMap,
    const llvm::DenseMap<Operation *, std::string> *opToHelper,
    const llvm::DenseMap<Operation *, std::vector<Value>> *opToInputs,
    const llvm::DenseMap<Operation *, std::vector<Operation *>>
        *opToCoAnchors) {
  // Get the for loop bounds: lower, upper, step are first 3 operands
  // scf.for %iv = %lb to %ub step %step iter_args(%arg = %init)
  Value lowerBound = op->getOperand(0);
  Value upperBound = op->getOperand(1);
  Value step = op->getOperand(2);

  // Get the induction variable from the region
  Region &bodyRegion = op->getRegion(0);
  Block &entryBlock = bodyRegion.front();

  // The induction variable is the first block argument
  Value inductionVar = entryBlock.getArgument(0);

  // Get iter_args - they start from operand 3
  unsigned numIterArgs = op->getNumOperands() - 3;

  // Map for loop results to iter_args
  // %107:3 = scf.for ... iter_args(%arg9, %arg10, %arg11)
  // means %107#0 -> %arg9, %107#1 -> %arg10, etc.
  if (argSubstitutionMap) {
    for (unsigned i = 0; i < op->getNumResults() && i < numIterArgs; ++i) {
      Value forResult = op->getResult(i);
      Value iterArg = entryBlock.getArgument(1 + i);
      (*argSubstitutionMap)[forResult] = iterArg;
    }
  }

  // Print iter_args initialization first
  for (unsigned i = 0; i < numIterArgs; ++i) {
    Value initValue = op->getOperand(3 + i);
    Value iterArg = entryBlock.getArgument(1 + i);

    for (unsigned j = 0; j < indent; ++j)
      os << "  ";
    os << getValueName(iterArg, argSubstitutionMap) << " = "
       << getValueName(initValue, argSubstitutionMap) << "\n";
  }

  // Print the for loop header
  for (unsigned i = 0; i < indent; ++i)
    os << "  ";
  std::string ivName = getValueName(inductionVar, argSubstitutionMap);
  os << "for " << ivName << " in range("
     << getValueName(lowerBound, argSubstitutionMap) << ", "
     << getValueName(upperBound, argSubstitutionMap) << ", "
     << getValueName(step, argSubstitutionMap) << "):\n";

  // Print the body
  printRegion(bodyRegion, os, opNameMap, allocInfoMap, skippedOps, indent + 1,
              argSubstitutionMap, {}, opToHelper, opToInputs, opToCoAnchors);
}

// Print scf.if with yield-to-assignment conversion
void printIfOp(
    Operation *op, llvm::raw_ostream &os,
    const llvm::StringMap<StringRef> &opNameMap,
    const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
    llvm::DenseSet<Operation *> &skippedOps, unsigned indent,
    DenseMap<Value, Value> *argSubstitutionMap,
    const llvm::DenseMap<Operation *, std::string> *opToHelper,
    const llvm::DenseMap<Operation *, std::vector<Value>> *opToInputs,
    const llvm::DenseMap<Operation *, std::vector<Operation *>>
        *opToCoAnchors) {
  // Get the condition operand
  Value condition = op->getOperand(0);

  // Map if's results to yield targets for subsequent use
  // (Like for loop, usages of if results after the if should refer to the
  // result) But for if, we keep the original result names

  // Get the if's results - these become the yield targets
  SmallVector<Value> ifResults;
  for (Value result : op->getResults()) {
    ifResults.push_back(result);
  }

  // Print "if condition:"
  for (unsigned i = 0; i < indent; ++i)
    os << "  ";
  os << "if " << getValueName(condition, argSubstitutionMap) << ":\n";

  // Print then region with yield targets
  if (op->getNumRegions() > 0) {
    printRegion(op->getRegion(0), os, opNameMap, allocInfoMap, skippedOps,
                indent + 1, argSubstitutionMap, ifResults, opToHelper,
                opToInputs, opToCoAnchors);
  }

  // Print else region if it exists and is non-empty
  if (op->getNumRegions() > 1 && !op->getRegion(1).empty()) {
    for (unsigned i = 0; i < indent; ++i)
      os << "  ";
    os << "else:\n";
    printRegion(op->getRegion(1), os, opNameMap, allocInfoMap, skippedOps,
                indent + 1, argSubstitutionMap, ifResults, opToHelper,
                opToInputs, opToCoAnchors);
  }
}

// Print warp_specialize operation in TLX async_tasks format
void printWarpSpecialize(
    Operation *op, llvm::raw_ostream &os,
    const llvm::StringMap<StringRef> &opNameMap,
    const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
    llvm::DenseSet<Operation *> &skippedOps, unsigned indent,
    const llvm::DenseMap<Operation *, std::string> *opToHelper = nullptr,
    const llvm::DenseMap<Operation *, std::vector<Value>> *opToInputs = nullptr,
    const llvm::DenseMap<Operation *, std::vector<Operation *>> *opToCoAnchors =
        nullptr) {
  // Print "with tlx.async_tasks():"
  for (unsigned i = 0; i < indent; ++i)
    os << "  ";
  os << "with tlx.async_tasks():\n";

  // Get the operands passed to warp_specialize
  SmallVector<Value> wsOperands;
  for (Value operand : op->getOperands()) {
    wsOperands.push_back(operand);
  }

  unsigned regionIdx = 0;
  for (Region &region : op->getRegions()) {
    if (regionIdx == 0) {
      // First region is the default clause
      // Build substitution map: region block args -> warp_specialize operands
      DenseMap<Value, Value> argSubstitutionMap;
      if (!region.empty()) {
        Block &entryBlock = region.front();
        for (unsigned i = 0;
             i < entryBlock.getNumArguments() && i < wsOperands.size(); ++i) {
          argSubstitutionMap[entryBlock.getArgument(i)] = wsOperands[i];
        }
      }

      // Print indentation and "with tlx.async_task("default"):"
      for (unsigned i = 0; i < indent + 1; ++i)
        os << "  ";
      os << "with tlx.async_task(\"default\"):\n";

      // Print region contents with extra indentation and substitution map
      printRegion(region, os, opNameMap, allocInfoMap, skippedOps, indent + 2,
                  &argSubstitutionMap, {}, opToHelper, opToInputs,
                  opToCoAnchors);
    } else {
      // Subsequent regions contain ttg.warp_specialize.partitions
      // which has multiple regions (one per partition)
      for (Block &block : region) {
        for (Operation &innerOp : block) {
          if (innerOp.getName().getStringRef() ==
              "ttg.warp_specialize.partitions") {
            // Each region in warp_specialize.partitions is a partition
            for (Region &partitionRegion : innerOp.getRegions()) {
              // Build substitution map for this partition
              DenseMap<Value, Value> argSubstitutionMap;
              if (!partitionRegion.empty()) {
                Block &entryBlock = partitionRegion.front();
                for (unsigned i = 0;
                     i < entryBlock.getNumArguments() && i < wsOperands.size();
                     ++i) {
                  argSubstitutionMap[entryBlock.getArgument(i)] = wsOperands[i];
                }
              }

              // Print indentation and "with tlx.async_task():"
              for (unsigned i = 0; i < indent + 1; ++i)
                os << "  ";
              os << "with tlx.async_task():\n";

              // Print partition contents
              printRegion(partitionRegion, os, opNameMap, allocInfoMap,
                          skippedOps, indent + 2, &argSubstitutionMap, {},
                          opToHelper, opToInputs, opToCoAnchors);
            }
          }
        }
      }
    }
    regionIdx++;
  }
}

// Print operation in simplified TLX format
void printSimplifiedOp(
    Operation *op, llvm::raw_ostream &os,
    const llvm::StringMap<StringRef> &opNameMap,
    const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap, unsigned indent,
    const DenseMap<Value, Value> *argSubstitutionMap = nullptr,
    const llvm::DenseMap<Operation *, std::string> *opToHelper = nullptr,
    const llvm::DenseMap<Operation *, std::vector<Value>> *opToInputs = nullptr,
    const llvm::DenseMap<Operation *, std::vector<Operation *>> *opToCoAnchors =
        nullptr) {
  StringRef opName = op->getName().getStringRef();

  // Print indentation
  for (unsigned i = 0; i < indent; ++i)
    os << "  ";

  // Check if this op should be printed as a helper call
  if (opToHelper && opToHelper->count(op)) {
    auto helperIt = opToHelper->find(op);
    if (helperIt != opToHelper->end()) {
      // Check if this is a multi-output pattern
      std::vector<Operation *> coAnchors;
      if (opToCoAnchors && opToCoAnchors->count(op)) {
        coAnchors = opToCoAnchors->find(op)->second;
      }

      // Print result(s) assignment
      if (!coAnchors.empty()) {
        // Multi-output: print all results
        os << getValueName(op->getResult(0), argSubstitutionMap);
        for (Operation *coAnchor : coAnchors) {
          os << ", "
             << getValueName(coAnchor->getResult(0), argSubstitutionMap);
        }
        os << " = ";
      } else if (op->getNumResults() > 0) {
        os << getValueName(op->getResult(0), argSubstitutionMap) << " = ";
      }

      // Print helper call
      os << helperIt->second << "(";
      if (opToInputs && opToInputs->count(op)) {
        auto inputsIt = opToInputs->find(op);
        if (inputsIt != opToInputs->end()) {
          const auto &inputs = inputsIt->second;
          for (size_t i = 0; i < inputs.size(); ++i) {
            if (i > 0)
              os << ", ";
            os << getValueName(inputs[i], argSubstitutionMap);
          }
        }
      }
      os << ")\n";
      return;
    }
  }

  // Special handling for arith.constant - print the value directly
  if (opName == "arith.constant") {
    if (op->getNumResults() > 0) {
      os << getValueName(op->getResult(0), argSubstitutionMap) << " = ";
    }
    if (auto valueAttr = op->getAttr("value")) {
      printConstantValue(valueAttr, os);
    } else {
      os << "const";
    }
    os << "\n";
    return;
  }

  // Special handling for local_alloc
  if (opName == "ttg.local_alloc") {
    auto it = allocInfoMap.find(op);
    if (it != allocInfoMap.end()) {
      const LocalAllocInfo &info = it->second;
      if (info.isBarrierAlloc) {
        // Print as result = tlx.alloc_barriers(count)
        if (op->getNumResults() > 0) {
          os << getValueName(op->getResult(0), argSubstitutionMap) << " = ";
        }
        os << "tlx.alloc_barriers(" << info.barrierCount << ")\n";
        return;
      } else {
        // Print as tlx.local_alloc((shape), dtype, count)
        if (op->getNumResults() > 0) {
          os << getValueName(op->getResult(0), argSubstitutionMap) << " = ";
        }
        os << "tlx.local_alloc((";
        for (size_t i = 0; i < info.shape.size(); ++i) {
          if (i > 0)
            os << ", ";
          os << info.shape[i];
        }
        os << "), " << getElementTypeName(info.elementType) << ", "
           << info.bufferCount << ")\n";
        return;
      }
    }
  }

  // Get the TLX name or use original
  auto it = opNameMap.find(opName);
  StringRef tlxName = (it != opNameMap.end()) ? it->second : opName;

  // Print results
  if (op->getNumResults() > 0) {
    if (op->getNumResults() == 1) {
      os << getValueName(op->getResult(0), argSubstitutionMap);
    } else {
      os << "(";
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        if (i > 0)
          os << ", ";
        os << getValueName(op->getResult(i), argSubstitutionMap);
      }
      os << ")";
    }
    os << " = ";
  }

  // Print operation name
  os << tlxName;

  // Print operands in parentheses
  os << "(";
  bool first = true;
  for (Value operand : op->getOperands()) {
    if (!first)
      os << ", ";
    first = false;
    os << getValueName(operand, argSubstitutionMap);
  }
  os << ")";

  os << "\n";
}

// Print a block
void printBlock(
    Block &block, llvm::raw_ostream &os,
    const llvm::StringMap<StringRef> &opNameMap,
    const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
    llvm::DenseSet<Operation *> &skippedOps, unsigned indent,
    DenseMap<Value, Value> *argSubstitutionMap = nullptr,
    ArrayRef<Value> yieldTargets = {},
    const llvm::DenseMap<Operation *, std::string> *opToHelper = nullptr,
    const llvm::DenseMap<Operation *, std::vector<Value>> *opToInputs = nullptr,
    const llvm::DenseMap<Operation *, std::vector<Operation *>> *opToCoAnchors =
        nullptr) {
  // Print block arguments if any
  if (!block.getArguments().empty() && !block.isEntryBlock()) {
    for (unsigned i = 0; i < indent; ++i)
      os << "  ";
    os << "^bb(";
    for (unsigned i = 0; i < block.getNumArguments(); ++i) {
      if (i > 0)
        os << ", ";
      os << getValueName(block.getArgument(i), argSubstitutionMap);
    }
    os << "):\n";
  }

  // Print operations
  for (Operation &op : block) {
    // Skip module and function ops - just print their contents
    if (isa<ModuleOp>(op)) {
      for (Region &region : op.getRegions()) {
        printRegion(region, os, opNameMap, allocInfoMap, skippedOps, indent,
                    argSubstitutionMap, {}, opToHelper, opToInputs,
                    opToCoAnchors);
      }
      continue;
    }

    if (auto funcOp = dyn_cast<tt::FuncOp>(op)) {
      os << "func " << funcOp.getName() << "(";
      // Print function arguments
      for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
        if (i > 0)
          os << ", ";
        os << getValueName(funcOp.getArgument(i), argSubstitutionMap);
      }
      os << ") {\n";
      for (Region &region : op.getRegions()) {
        printRegion(region, os, opNameMap, allocInfoMap, skippedOps, indent + 1,
                    argSubstitutionMap, {}, opToHelper, opToInputs,
                    opToCoAnchors);
      }
      os << "}\n";
      continue;
    }

    // Check if we should skip this operation
    if (shouldSkipOp(&op, allocInfoMap, skippedOps)) {
      continue;
    }

    // Check if this is a co-anchor that was already printed with primary anchor
    if (opToCoAnchors) {
      bool isCoAnchor = false;
      for (auto &kv : *opToCoAnchors) {
        for (Operation *coAnchor : kv.second) {
          if (coAnchor == &op) {
            isCoAnchor = true;
            break;
          }
        }
        if (isCoAnchor)
          break;
      }
      if (isCoAnchor)
        continue;
    }

    // Special handling for scf.yield - convert to assignments if we have yield
    // targets, otherwise skip entirely
    if (op.getName().getStringRef() == "scf.yield") {
      if (!yieldTargets.empty()) {
        // Print assignments: yieldTarget = yieldOperand
        for (unsigned i = 0; i < op.getNumOperands() && i < yieldTargets.size();
             ++i) {
          for (unsigned j = 0; j < indent; ++j)
            os << "  ";
          os << getValueName(yieldTargets[i], argSubstitutionMap) << " = "
             << getValueName(op.getOperand(i), argSubstitutionMap) << "\n";
        }
      }
      // Skip yield in TLX output (either handled above or just skip)
      continue;
    }

    // Special handling for warp_specialize
    if (op.getName().getStringRef() == "ttg.warp_specialize") {
      printWarpSpecialize(&op, os, opNameMap, allocInfoMap, skippedOps, indent,
                          opToHelper, opToInputs, opToCoAnchors);
      continue;
    }

    // Special handling for scf.for - Python range syntax
    if (op.getName().getStringRef() == "scf.for") {
      printForOp(&op, os, opNameMap, allocInfoMap, skippedOps, indent,
                 argSubstitutionMap, opToHelper, opToInputs, opToCoAnchors);
      continue;
    }

    // Special handling for scf.if - Python if/else with yield-to-assignment
    if (op.getName().getStringRef() == "scf.if") {
      printIfOp(&op, os, opNameMap, allocInfoMap, skippedOps, indent,
                argSubstitutionMap, opToHelper, opToInputs, opToCoAnchors);
      continue;
    }

    // Handle operations with regions (while, etc.)
    if (op.getNumRegions() > 0) {
      printSimplifiedOp(&op, os, opNameMap, allocInfoMap, indent,
                        argSubstitutionMap, opToHelper, opToInputs,
                        opToCoAnchors);
      // Print indentation and opening brace
      for (unsigned i = 0; i < indent; ++i)
        os << "  ";
      os << "{\n";
      for (Region &region : op.getRegions()) {
        printRegion(region, os, opNameMap, allocInfoMap, skippedOps, indent + 1,
                    argSubstitutionMap, {}, opToHelper, opToInputs,
                    opToCoAnchors);
      }
      for (unsigned i = 0; i < indent; ++i)
        os << "  ";
      os << "}\n";
    } else {
      printSimplifiedOp(&op, os, opNameMap, allocInfoMap, indent,
                        argSubstitutionMap, opToHelper, opToInputs,
                        opToCoAnchors);
    }
  }
}

void printRegion(
    Region &region, llvm::raw_ostream &os,
    const llvm::StringMap<StringRef> &opNameMap,
    const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
    llvm::DenseSet<Operation *> &skippedOps, unsigned indent,
    DenseMap<Value, Value> *argSubstitutionMap, ArrayRef<Value> yieldTargets,
    const llvm::DenseMap<Operation *, std::string> *opToHelper,
    const llvm::DenseMap<Operation *, std::vector<Value>> *opToInputs,
    const llvm::DenseMap<Operation *, std::vector<Operation *>>
        *opToCoAnchors) {
  for (Block &block : region) {
    printBlock(block, os, opNameMap, allocInfoMap, skippedOps, indent,
               argSubstitutionMap, yieldTargets, opToHelper, opToInputs,
               opToCoAnchors);
  }
}

} // namespace

struct TLXPrintTTGIRToTLXPass
    : public impl::TLXPrintTTGIRToTLXBase<TLXPrintTTGIRToTLXPass> {
public:
  using impl::TLXPrintTTGIRToTLXBase<
      TLXPrintTTGIRToTLXPass>::TLXPrintTTGIRToTLXBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Build the lookup map
    static llvm::StringMap<StringRef> opNameMap = buildOpNameMap();

    // ================================================================
    // General Pattern Discovery: Walk ALL ops, merge overlapping patterns
    // ================================================================
    constexpr size_t MIN_PATTERN_OPS = 4;
    constexpr size_t MAX_PATTERN_OPS = 12;

    // Skip trivial/control-flow ops that shouldn't be pattern roots
    auto shouldSkipOp = [](Operation *op) {
      StringRef name = op->getName().getStringRef();
      return name == "arith.constant" || name.starts_with("scf.") ||
             name.starts_with("cf.") || name == "tt.splat" ||
             name == "tt.make_range" || name == "ttg.warp_yield" ||
             name == "ttg.warp_return" || name.starts_with("func.") ||
             name == "tt.return" || name == "builtin.module";
    };

    // Build signature by following def chains from root op
    auto buildSig =
        [&shouldSkipOp](Operation *root, std::vector<Operation *> &patternOps,
                        llvm::DenseMap<Operation *, size_t> &opToIdx,
                        size_t maxOps) {
          patternOps.clear();
          opToIdx.clear();
          std::vector<Operation *> worklist;
          std::ostringstream sig;

          patternOps.push_back(root);
          opToIdx[root] = 0;
          worklist.push_back(root);

          size_t processed = 0;
          while (processed < worklist.size() && patternOps.size() < maxOps) {
            Operation *op = worklist[processed++];
            for (Value operand : op->getOperands()) {
              if (Operation *defOp = operand.getDefiningOp()) {
                if (opToIdx.count(defOp))
                  continue;
                if (shouldSkipOp(defOp))
                  continue;
                size_t idx = patternOps.size();
                patternOps.push_back(defOp);
                opToIdx[defOp] = idx;
                worklist.push_back(defOp);
              }
            }
          }

          // Build canonical signature: op_name(dependency_refs)
          for (size_t i = 0; i < patternOps.size(); ++i) {
            Operation *op = patternOps[i];
            sig << i << ":" << op->getName().getStringRef().str() << "(";
            for (unsigned j = 0; j < op->getNumOperands(); ++j) {
              if (j > 0)
                sig << ",";
              Value operand = op->getOperand(j);
              if (Operation *defOp = operand.getDefiningOp()) {
                auto it = opToIdx.find(defOp);
                if (it != opToIdx.end())
                  sig << "op" << it->second;
                else
                  sig << "ext";
              } else {
                sig << "arg";
              }
            }
            sig << ")";
          }
          return sig.str();
        };

    // Check if pattern is trivial (mostly type conversions)
    auto isTrivial = [](const std::vector<Operation *> &ops) {
      size_t trivialCnt = 0;
      for (Operation *op : ops) {
        StringRef name = op->getName().getStringRef();
        if (name == "arith.trunci" || name == "arith.extui" ||
            name == "arith.extsi" || name == "arith.andi" ||
            name == "arith.ori" || name == "arith.xori")
          trivialCnt++;
      }
      return trivialCnt > ops.size() / 2;
    };

    // Data structures for pattern discovery
    // sig -> list of root ops with that signature
    std::map<std::string, std::vector<Operation *>> sigToRoots;
    // sig -> list of pattern ops for each occurrence
    std::map<std::string, std::vector<std::vector<Operation *>>>
        sigToPatternOps;

    // Pass 1: Discover patterns from ALL ops
    m.walk([&](Operation *op) {
      if (shouldSkipOp(op))
        return;
      if (op->getNumResults() == 0)
        return; // Only consider ops with results

      std::vector<Operation *> patternOps;
      llvm::DenseMap<Operation *, size_t> opToIdx;
      std::string sig = buildSig(op, patternOps, opToIdx, MAX_PATTERN_OPS);

      if (patternOps.size() >= MIN_PATTERN_OPS && !isTrivial(patternOps)) {
        sigToRoots[sig].push_back(op);
        sigToPatternOps[sig].push_back(patternOps);
      }
    });

    // Pass 1.5: Filter out subset patterns
    // A pattern A is a strict subset of pattern B if:
    // 1. They have the same occurrence count
    // 2. B has strictly more ops than A
    // 3. For EVERY occurrence, ALL of A's ops are contained in B's ops
    // 4. A's root is NOT a root of B (it's an internal op of B)
    std::set<std::string> subsetSigs;

    // Also track: for each pattern, which patterns have it as an internal op
    // This helps identify patterns that will be "absorbed" by larger patterns
    for (auto &kvA : sigToRoots) {
      const std::string &sigA = kvA.first;
      if (kvA.second.size() < 2)
        continue;

      size_t occA = kvA.second.size();
      size_t sizeA = sigToPatternOps[sigA][0].size();

      for (auto &kvB : sigToRoots) {
        const std::string &sigB = kvB.first;
        if (sigA == sigB)
          continue;
        if (kvB.second.size() < 2)
          continue;

        size_t occB = kvB.second.size();
        size_t sizeB = sigToPatternOps[sigB][0].size();

        // Must have same occurrence count and B must be strictly larger
        if (occA != occB || sizeB <= sizeA)
          continue;

        // Check if A is a subset of B across ALL occurrences
        bool isSubsetAllOccurrences = true;
        for (size_t i = 0; i < occA && isSubsetAllOccurrences; ++i) {
          // Build set of B's ops for this occurrence
          llvm::DenseSet<Operation *> opsBSet;
          for (Operation *op : sigToPatternOps[sigB][i]) {
            opsBSet.insert(op);
          }

          // Check if ALL of A's ops are in B
          for (Operation *opA : sigToPatternOps[sigA][i]) {
            if (!opsBSet.count(opA)) {
              isSubsetAllOccurrences = false;
              break;
            }
          }

          // Also verify A's root is NOT the root of B (otherwise they're the
          // same pattern starting point)
          if (isSubsetAllOccurrences) {
            Operation *rootA = kvA.second[i];
            Operation *rootB = kvB.second[i];
            if (rootA == rootB) {
              isSubsetAllOccurrences = false;
            }
          }
        }

        if (isSubsetAllOccurrences) {
          subsetSigs.insert(sigA);
          break;
        }
      }
    }

    // Additional check: if A's root is an internal (non-root) op of B for ALL
    // occurrences, then A should be absorbed even if A's ops aren't fully
    // contained in B (because A's root will be skipped when B is processed)
    for (auto &kvA : sigToRoots) {
      const std::string &sigA = kvA.first;
      if (subsetSigs.count(sigA))
        continue; // Already marked
      if (kvA.second.size() < 2)
        continue;

      size_t occA = kvA.second.size();

      for (auto &kvB : sigToRoots) {
        const std::string &sigB = kvB.first;
        if (sigA == sigB)
          continue;
        if (kvB.second.size() != occA)
          continue;
        if (kvB.second.size() < 2)
          continue;

        // Check if A's root is a non-root op of B for ALL occurrences
        bool rootAbsorbedAllOccurrences = true;
        for (size_t i = 0; i < occA && rootAbsorbedAllOccurrences; ++i) {
          Operation *rootA = kvA.second[i];
          Operation *rootB = kvB.second[i];

          // A's root must not be B's root
          if (rootA == rootB) {
            rootAbsorbedAllOccurrences = false;
            continue;
          }

          // Check if A's root is in B's ops (as a non-root)
          auto &opsB = sigToPatternOps[sigB][i];
          bool foundInB = false;
          for (size_t j = 1; j < opsB.size(); ++j) { // Skip root (index 0)
            if (opsB[j] == rootA) {
              foundInB = true;
              break;
            }
          }
          if (!foundInB) {
            rootAbsorbedAllOccurrences = false;
          }
        }

        if (rootAbsorbedAllOccurrences) {
          subsetSigs.insert(sigA);
          break;
        }
      }
    }

    // Pass 2: Find overlapping patterns and merge them
    // Two patterns overlap if they share ops. We group patterns by shared ops.
    // For each pair of signatures with 2+ occurrences, check if their patterns
    // share ops across ALL occurrences consistently.

    // Map: op -> which signature's pattern it belongs to (for first occurrence)
    llvm::DenseMap<Operation *, std::string> opToSig;
    for (auto &kv : sigToPatternOps) {
      if (kv.second.size() < 2)
        continue;
      for (Operation *op : kv.second[0]) {
        opToSig[op] = kv.first;
      }
    }

    // Find signatures that share ops
    std::map<std::set<std::string>, std::vector<std::string>> sharedOpsGroups;
    for (auto &kv : sigToPatternOps) {
      const std::string &sig = kv.first;
      if (kv.second.size() < 2)
        continue;

      std::set<std::string> sharedWith;
      sharedWith.insert(sig);

      for (Operation *op : kv.second[0]) {
        auto it = opToSig.find(op);
        if (it != opToSig.end() && it->second != sig) {
          // This op is shared with another pattern
          // Check if they share consistently across all occurrences
          const std::string &otherSig = it->second;
          if (sigToPatternOps[otherSig].size() == kv.second.size()) {
            sharedWith.insert(otherSig);
          }
        }
      }
      if (sharedWith.size() > 1) {
        sharedOpsGroups[sharedWith].push_back(sig);
      }
    }

    // Merge patterns that share ops into multi-output patterns
    std::set<std::string> mergedSigs;
    // mergedGroup -> list of signatures in that group
    std::vector<std::vector<std::string>> mergedGroups;

    for (auto &kv : sharedOpsGroups) {
      auto &group = kv.first;
      if (group.size() >= 2) {
        std::vector<std::string> groupVec(group.begin(), group.end());
        // Verify all have same occurrence count
        size_t cnt = sigToRoots[groupVec[0]].size();
        bool valid = true;
        for (const auto &sig : groupVec) {
          if (sigToRoots[sig].size() != cnt) {
            valid = false;
            break;
          }
        }
        if (valid && cnt >= 2) {
          mergedGroups.push_back(groupVec);
          for (const auto &sig : groupVec) {
            mergedSigs.insert(sig);
          }
        }
      }
    }

    // Output data structures
    llvm::DenseMap<Operation *, std::string> opToHelper;
    llvm::DenseMap<Operation *, std::vector<Value>> opToInputs;
    llvm::DenseMap<Operation *, std::vector<Operation *>> opToCoAnchors;
    llvm::DenseSet<Operation *> opsToSkip;

    // ================================================================
    // Pre-compute which ops will be skipped
    // Only skip ops that are NON-ROOT ops in patterns that will be used
    // ================================================================
    llvm::DenseSet<Operation *> potentialOpsToSkip;

    // First, identify patterns that are strict subsets of larger patterns
    // A pattern A is a subset if ALL its roots are non-root ops in a larger
    // pattern B AND both A and B have the same occurrence count
    std::set<std::string> subsetToRemove;

    for (auto &kvA : sigToRoots) {
      const std::string &sigA = kvA.first;
      if (kvA.second.size() < 2)
        continue;
      if (subsetSigs.count(sigA))
        continue;

      size_t sizeA = sigToPatternOps[sigA][0].size();
      size_t occA = kvA.second.size();

      // Check if all roots of A are contained in some larger pattern B
      for (auto &kvB : sigToRoots) {
        const std::string &sigB = kvB.first;
        if (sigA == sigB)
          continue;
        if (kvB.second.size() < 2)
          continue;
        if (kvB.second.size() != occA)
          continue; // Must have same occurrence count

        size_t sizeB = sigToPatternOps[sigB][0].size();
        if (sizeB <= sizeA)
          continue; // B must be strictly larger

        // Check if ALL roots of A (across all occurrences) are non-root ops in
        // B for the SAME occurrence index
        bool allRootsInB = true;
        for (size_t i = 0; i < occA && allRootsInB; ++i) {
          Operation *rootA = kvA.second[i];

          // Get B's ops for this occurrence
          auto &opsB = sigToPatternOps[sigB][i];
          llvm::DenseSet<Operation *> opsBSet(opsB.begin(), opsB.end());

          // rootA must be in B's ops but NOT be any root of B
          // For merged patterns, B might have multiple roots
          if (!opsBSet.count(rootA)) {
            allRootsInB = false;
          } else {
            // Check if rootA is a root of B (not just the primary root)
            // B's root for this occurrence is kvB.second[i], but for merged
            // patterns, check if rootA is the root of any signature in the
            // same merged group
            bool isRootOfB = (rootA == kvB.second[i]);
            if (isRootOfB) {
              allRootsInB = false;
            }
          }
        }

        if (allRootsInB) {
          subsetToRemove.insert(sigA);
          break;
        }
      }
    }

    // Collect all non-root ops from merged patterns
    for (auto &group : mergedGroups) {
      // Check if any sig in group is in subsetToRemove
      bool skipGroup = false;
      for (const auto &sig : group) {
        if (subsetToRemove.count(sig)) {
          skipGroup = true;
          break;
        }
      }
      if (skipGroup)
        continue;

      size_t numOccurrences = sigToRoots[group[0]].size();
      for (size_t i = 0; i < numOccurrences; ++i) {
        llvm::DenseSet<Operation *> rootsThisOccurrence;
        for (const auto &sig : group) {
          rootsThisOccurrence.insert(sigToRoots[sig][i]);
        }
        for (const auto &sig : group) {
          for (Operation *op : sigToPatternOps[sig][i]) {
            if (!rootsThisOccurrence.count(op)) {
              potentialOpsToSkip.insert(op);
            }
          }
        }
      }
    }

    // Collect all non-root ops from non-merged patterns
    for (auto &kv : sigToRoots) {
      const std::string &sig = kv.first;
      if (mergedSigs.count(sig))
        continue;
      if (subsetToRemove.count(sig))
        continue;

      auto &roots = kv.second;
      if (roots.size() < 2)
        continue;

      for (size_t i = 0; i < roots.size(); ++i) {
        auto &patternOps = sigToPatternOps[sig][i];
        for (size_t j = 1; j < patternOps.size(); ++j) {
          potentialOpsToSkip.insert(patternOps[j]);
        }
      }
    }

    // Now filter: patterns are valid if ALL their roots are NOT in
    // potentialOpsToSkip
    std::set<std::string> validSigs;
    for (auto &kv : sigToRoots) {
      const std::string &sig = kv.first;
      if (kv.second.size() < 2)
        continue;
      if (subsetToRemove.count(sig))
        continue;

      bool allRootsValid = true;
      for (Operation *root : kv.second) {
        if (potentialOpsToSkip.count(root)) {
          allRootsValid = false;
          break;
        }
      }
      if (allRootsValid) {
        validSigs.insert(sig);
      }
    }

    // Filter merged groups to only include valid ones
    // Filter merged groups: ALL roots across ALL signatures must NOT be in
    // potentialOpsToSkip
    std::vector<std::vector<std::string>> validMergedGroups;
    for (auto &group : mergedGroups) {
      bool allValid = true;

      // First check if all sigs are in validSigs
      for (const auto &sig : group) {
        if (!validSigs.count(sig)) {
          allValid = false;
          break;
        }
      }
      if (!allValid)
        continue;

      // Now check if ALL roots from ALL signatures are NOT in
      // potentialOpsToSkip
      size_t numOccurrences = sigToRoots[group[0]].size();
      for (size_t i = 0; i < numOccurrences && allValid; ++i) {
        for (const auto &sig : group) {
          Operation *root = sigToRoots[sig][i];
          if (potentialOpsToSkip.count(root)) {
            allValid = false;
            break;
          }
        }
      }

      if (allValid) {
        validMergedGroups.push_back(group);
      }
    }

    size_t helperId = 0;

    // Track which groups/sigs got assigned a helper name (for printing later)
    std::map<size_t, std::string> mergedGroupIdxToHelper;
    std::map<std::string, std::string> singleSigToHelper;

    // Register merged multi-output patterns (only valid ones)
    // Sort validMergedGroups by total op count (descending) so larger patterns
    // are registered first
    std::sort(validMergedGroups.begin(), validMergedGroups.end(),
              [&](const std::vector<std::string> &a,
                  const std::vector<std::string> &b) {
                // Calculate total ops for group a
                size_t opsA = 0;
                llvm::DenseSet<Operation *> seenA;
                for (const auto &sig : a) {
                  for (Operation *op : sigToPatternOps[sig][0]) {
                    if (!seenA.count(op)) {
                      seenA.insert(op);
                      opsA++;
                    }
                  }
                }
                // Calculate total ops for group b
                size_t opsB = 0;
                llvm::DenseSet<Operation *> seenB;
                for (const auto &sig : b) {
                  for (Operation *op : sigToPatternOps[sig][0]) {
                    if (!seenB.count(op)) {
                      seenB.insert(op);
                      opsB++;
                    }
                  }
                }
                return opsA > opsB; // Larger patterns first
              });

    for (size_t groupIdx = 0; groupIdx < validMergedGroups.size(); ++groupIdx) {
      auto &group = validMergedGroups[groupIdx];
      size_t numOccurrences = sigToRoots[group[0]].size();

      // First pass: collect valid occurrences (those not conflicting)
      struct OccurrenceData {
        Operation *primary;
        std::vector<Operation *> coAnchors;
        std::vector<Operation *> combinedOps;
        std::vector<Value> extInputs;
      };
      std::vector<OccurrenceData> validOccurrences;

      for (size_t i = 0; i < numOccurrences; ++i) {
        // Collect all roots for this occurrence
        std::vector<Operation *> rootsThisOccurrence;
        for (const auto &sig : group) {
          rootsThisOccurrence.push_back(sigToRoots[sig][i]);
        }

        // Skip this occurrence if the primary root is already registered or
        // skipped
        Operation *primary = rootsThisOccurrence[0];
        if (opToHelper.count(primary) || opsToSkip.count(primary)) {
          continue;
        }

        // Also skip if ANY co-anchor is already registered OR in opsToSkip
        // (would cause conflict)
        bool anyCoAnchorConflict = false;
        for (size_t j = 1; j < rootsThisOccurrence.size(); ++j) {
          if (opToHelper.count(rootsThisOccurrence[j]) ||
              opsToSkip.count(rootsThisOccurrence[j])) {
            anyCoAnchorConflict = true;
            break;
          }
        }
        if (anyCoAnchorConflict) {
          continue;
        }

        // Build combined pattern ops (union)
        std::vector<Operation *> combinedOps;
        llvm::DenseSet<Operation *> seenOps;
        for (const auto &sig : group) {
          for (Operation *op : sigToPatternOps[sig][i]) {
            if (!seenOps.count(op)) {
              seenOps.insert(op);
              combinedOps.push_back(op);
            }
          }
        }

        // Build opToIdx
        llvm::DenseMap<Operation *, size_t> opToIdx;
        for (size_t k = 0; k < combinedOps.size(); ++k)
          opToIdx[combinedOps[k]] = k;

        // Collect external inputs
        std::vector<Value> extInputs;
        llvm::DenseSet<Value> seen;
        for (Operation *op : combinedOps) {
          for (Value operand : op->getOperands()) {
            Operation *defOp = operand.getDefiningOp();
            if (!defOp || !opToIdx.count(defOp)) {
              if (!seen.count(operand)) {
                seen.insert(operand);
                extInputs.push_back(operand);
              }
            }
          }
        }

        std::vector<Operation *> coAnchors(rootsThisOccurrence.begin() + 1,
                                           rootsThisOccurrence.end());
        validOccurrences.push_back(
            {primary, coAnchors, combinedOps, extInputs});
      }

      // Only create helper if we have at least one valid occurrence
      if (validOccurrences.empty()) {
        continue;
      }

      // Filter occurrences: keep only those with the same number of external
      // inputs as the first occurrence (ensures consistent function signature)
      size_t expectedInputCount = validOccurrences[0].extInputs.size();
      std::vector<OccurrenceData> consistentOccurrences;
      for (auto &occ : validOccurrences) {
        if (occ.extInputs.size() == expectedInputCount) {
          consistentOccurrences.push_back(std::move(occ));
        }
      }

      // Need at least 1 consistent occurrence to create a pattern
      if (consistentOccurrences.empty()) {
        continue;
      }

      std::string helperName = "_auto_pattern_" + std::to_string(helperId++);
      mergedGroupIdxToHelper[groupIdx] = helperName;

      // Second pass: register all consistent occurrences
      for (auto &occ : consistentOccurrences) {
        opToHelper[occ.primary] = helperName;
        opToInputs[occ.primary] = occ.extInputs;
        opToCoAnchors[occ.primary] = occ.coAnchors;

        for (Operation *coAnchor : occ.coAnchors) {
          opToHelper[coAnchor] = helperName;
          opToInputs[coAnchor] = occ.extInputs;
        }

        // Mark non-root ops for skipping
        llvm::DenseSet<Operation *> rootsThisOcc;
        rootsThisOcc.insert(occ.primary);
        for (Operation *coAnchor : occ.coAnchors)
          rootsThisOcc.insert(coAnchor);

        for (Operation *op : occ.combinedOps) {
          if (!rootsThisOcc.count(op)) {
            opsToSkip.insert(op);
          }
        }
      }
    }

    // Register single-output patterns (only valid ones)
    for (auto &kv : sigToRoots) {
      const std::string &sig = kv.first;
      if (mergedSigs.count(sig))
        continue;
      if (subsetSigs.count(sig))
        continue;
      if (!validSigs.count(sig))
        continue;

      auto &roots = kv.second;
      if (roots.size() < 2)
        continue;

      // First pass: collect valid occurrences (those whose root is not already
      // registered or skipped)
      struct SingleOccurrenceData {
        Operation *root;
        std::vector<Operation *> patternOps;
        std::vector<Value> extInputs;
      };
      std::vector<SingleOccurrenceData> validOccurrences;

      for (size_t i = 0; i < roots.size(); ++i) {
        Operation *root = roots[i];
        // Skip this occurrence if the root is already registered or skipped
        if (opToHelper.count(root) || opsToSkip.count(root)) {
          continue;
        }

        auto &patternOps = sigToPatternOps[sig][i];

        llvm::DenseMap<Operation *, size_t> opToIdx;
        for (size_t j = 0; j < patternOps.size(); ++j)
          opToIdx[patternOps[j]] = j;

        std::vector<Value> extInputs;
        llvm::DenseSet<Value> seen;
        for (Operation *op : patternOps) {
          for (Value operand : op->getOperands()) {
            Operation *defOp = operand.getDefiningOp();
            if (!defOp || !opToIdx.count(defOp)) {
              if (!seen.count(operand)) {
                seen.insert(operand);
                extInputs.push_back(operand);
              }
            }
          }
        }

        validOccurrences.push_back({root, patternOps, extInputs});
      }

      // Only create helper if we have at least 2 valid occurrences
      if (validOccurrences.size() < 2) {
        continue;
      }

      // Filter occurrences: keep only those with the same number of external
      // inputs as the first occurrence (ensures consistent function signature)
      size_t expectedInputCount = validOccurrences[0].extInputs.size();
      std::vector<SingleOccurrenceData> consistentOccurrences;
      for (auto &occ : validOccurrences) {
        if (occ.extInputs.size() == expectedInputCount) {
          consistentOccurrences.push_back(std::move(occ));
        }
      }

      // Need at least 2 consistent occurrences to create a pattern
      if (consistentOccurrences.size() < 2) {
        continue;
      }

      std::string helperName = "_auto_pattern_" + std::to_string(helperId++);
      singleSigToHelper[sig] = helperName;

      // Second pass: register all consistent occurrences
      for (auto &occ : consistentOccurrences) {
        opToHelper[occ.root] = helperName;
        opToInputs[occ.root] = occ.extInputs;

        // Mark non-root ops for skipping
        for (size_t j = 1; j < occ.patternOps.size(); ++j) {
          opsToSkip.insert(occ.patternOps[j]);
        }
      }
    }

    // ================================================================
    // Call-Sequence Merging: detect consecutive calls to same helper
    // that ALWAYS appear together (e.g., always in pairs)
    // ================================================================
    std::map<std::string, std::vector<std::vector<Operation *>>>
        helperToCallSequences;

    m.walk([&](Block *block) {
      std::vector<Operation *> currentSeq;
      std::string currentHelper;
      for (Operation &op : *block) {
        auto it = opToHelper.find(&op);
        if (it != opToHelper.end()) {
          const std::string &helper = it->second;
          if (currentHelper == helper) {
            currentSeq.push_back(&op);
          } else {
            if (currentSeq.size() >= 2) {
              helperToCallSequences[currentHelper].push_back(currentSeq);
            }
            currentSeq.clear();
            currentSeq.push_back(&op);
            currentHelper = helper;
          }
        } else {
          if (currentSeq.size() >= 2) {
            helperToCallSequences[currentHelper].push_back(currentSeq);
          }
          currentSeq.clear();
          currentHelper.clear();
        }
      }
      if (currentSeq.size() >= 2) {
        helperToCallSequences[currentHelper].push_back(currentSeq);
      }
    });

    // Find helpers that ALWAYS appear in sequences of same length
    for (auto &kv : helperToCallSequences) {
      auto &sequences = kv.second;
      if (sequences.empty())
        continue;

      std::map<size_t, size_t> lengthCounts;
      for (auto &seq : sequences)
        lengthCounts[seq.size()]++;

      size_t dominantLen = 0;
      size_t totalSeqs = sequences.size();
      for (auto &lc : lengthCounts) {
        if (lc.second == totalSeqs && lc.first >= 2) {
          dominantLen = lc.first;
          break;
        }
      }

      if (dominantLen >= 2) {
        for (auto &seq : sequences) {
          Operation *primary = seq[0];
          std::vector<Operation *> coAnchors(seq.begin() + 1, seq.end());
          if (opToCoAnchors.count(primary)) {
            auto &existing = opToCoAnchors[primary];
            existing.insert(existing.end(), coAnchors.begin(), coAnchors.end());
          } else {
            opToCoAnchors[primary] = coAnchors;
          }
        }
      }
    }

    // ================================================================
    // Print helper definitions
    // ================================================================
    // Filter opToHelper: remove any entries whose root op is in opsToSkip
    // This happens when a pattern's root is an internal op of another pattern
    std::vector<Operation *> toRemove;
    for (auto &kv : opToHelper) {
      if (opsToSkip.count(kv.first)) {
        toRemove.push_back(kv.first);
      }
    }
    for (Operation *op : toRemove) {
      opToHelper.erase(op);
      opToInputs.erase(op);
      opToCoAnchors.erase(op);
    }

    // Now collect which helper names are actually used
    // Build a set of all co-anchors (these don't generate calls, only primaries
    // do)
    llvm::DenseSet<Operation *> allCoAnchors;
    for (auto &kv : opToCoAnchors) {
      for (Operation *coAnchor : kv.second) {
        allCoAnchors.insert(coAnchor);
      }
    }

    std::set<std::string> usedHelperNames;
    std::map<std::string, size_t> helperCallCount;
    for (auto &kv : opToHelper) {
      usedHelperNames.insert(kv.second);
      // Only count primary roots (not co-anchors) since co-anchors don't
      // generate separate calls
      if (!allCoAnchors.count(kv.first)) {
        helperCallCount[kv.second]++;
      }
    }

    if (helperId > 0) {
      // First, collect all patterns with their ops for subset detection
      struct PatternInfo {
        std::string name;
        std::vector<std::vector<std::string>> sigs; // For merged groups
        std::string singleSig;                      // For single patterns
        llvm::DenseSet<Operation *> ops;
        size_t numOccurrences;
        bool isMerged;
      };
      std::vector<PatternInfo> allPatterns;

      size_t tempId = 0;
      // Collect merged patterns
      for (auto &group : validMergedGroups) {
        PatternInfo info;
        info.name = "_auto_pattern_" + std::to_string(tempId++);
        info.isMerged = true;
        info.numOccurrences = sigToRoots[group[0]].size();
        for (const auto &sig : group) {
          info.sigs.push_back({sig});
          for (Operation *op : sigToPatternOps[sig][0]) {
            info.ops.insert(op);
          }
        }
        allPatterns.push_back(std::move(info));
      }

      // Collect single-output patterns
      for (auto &kv : sigToRoots) {
        const std::string &sig = kv.first;
        if (mergedSigs.count(sig))
          continue;
        if (subsetToRemove.count(sig))
          continue;
        if (!validSigs.count(sig))
          continue;

        auto &roots = kv.second;
        if (roots.size() < 2)
          continue;

        PatternInfo info;
        info.name = "_auto_pattern_" + std::to_string(tempId++);
        info.isMerged = false;
        info.singleSig = sig;
        info.numOccurrences = roots.size();
        for (Operation *op : sigToPatternOps[sig][0]) {
          info.ops.insert(op);
        }
        allPatterns.push_back(std::move(info));
      }

      // Detect subset patterns: if pattern A's ops are entirely contained in
      // pattern B's ops and A has fewer ops, then A is a subset
      std::set<size_t> subsetPatternIndices;
      for (size_t i = 0; i < allPatterns.size(); ++i) {
        for (size_t j = 0; j < allPatterns.size(); ++j) {
          if (i == j)
            continue;
          // Check if pattern i is a subset of pattern j
          if (allPatterns[i].ops.size() >= allPatterns[j].ops.size())
            continue;
          if (allPatterns[i].numOccurrences != allPatterns[j].numOccurrences)
            continue;

          bool isSubset = true;
          for (Operation *op : allPatterns[i].ops) {
            if (!allPatterns[j].ops.count(op)) {
              isSubset = false;
              break;
            }
          }
          if (isSubset) {
            subsetPatternIndices.insert(i);
            break;
          }
        }
      }

      llvm::outs() << "# ========================================\n";
      llvm::outs() << "# Auto-discovered Helper Functions\n";
      llvm::outs() << "# ========================================\n\n";

      // Create a map from helper name to pattern info for printing
      // This ensures we only print helpers that are actually registered
      struct PrintablePattern {
        std::string helperName;
        std::vector<Operation *> ops;
        std::vector<Operation *> rootOps;
        size_t numOccurrences;
        bool isMerged;
      };
      std::vector<PrintablePattern> patternsToPrint;

      // Collect merged patterns that are actually used
      for (size_t groupIdx = 0; groupIdx < validMergedGroups.size();
           ++groupIdx) {
        // Skip groups that didn't get a helper name assigned
        auto it = mergedGroupIdxToHelper.find(groupIdx);
        if (it == mergedGroupIdxToHelper.end())
          continue;

        std::string helperName = it->second;
        auto &group = validMergedGroups[groupIdx];

        // Skip if not used (should always be used if we're here, but
        // double-check)
        if (!usedHelperNames.count(helperName))
          continue;

        // Skip if this pattern has 0 actual calls (all occurrences were
        // filtered out)
        if (helperCallCount[helperName] == 0)
          continue;

        // Skip if this pattern has 0 actual calls (all occurrences were
        // filtered out)
        if (helperCallCount[helperName] == 0)
          continue;

        PrintablePattern pp;
        pp.helperName = helperName;
        pp.isMerged = true;
        pp.numOccurrences =
            helperCallCount[helperName]; // Use actual call count

        // Combine ops
        llvm::DenseSet<Operation *> seenOps;
        for (const auto &sig : group) {
          for (Operation *op : sigToPatternOps[sig][0]) {
            if (!seenOps.count(op)) {
              seenOps.insert(op);
              pp.ops.push_back(op);
            }
          }
          pp.rootOps.push_back(sigToRoots[sig][0]);
        }
        patternsToPrint.push_back(std::move(pp));
      }

      // Collect single-output patterns that are actually used
      for (auto &kv : sigToRoots) {
        const std::string &sig = kv.first;
        if (mergedSigs.count(sig))
          continue;
        if (subsetSigs.count(sig))
          continue;
        if (!validSigs.count(sig))
          continue;
        if (kv.second.size() < 2)
          continue;

        // Skip sigs that didn't get a helper name assigned
        auto it = singleSigToHelper.find(sig);
        if (it == singleSigToHelper.end())
          continue;

        std::string helperName = it->second;

        // Skip if not used (should always be used if we're here, but
        // double-check)
        if (!usedHelperNames.count(helperName))
          continue;

        PrintablePattern pp;
        pp.helperName = helperName;
        pp.isMerged = false;
        pp.numOccurrences =
            helperCallCount[helperName]; // Use actual call count
        pp.ops = sigToPatternOps[sig][0];
        pp.rootOps.push_back(kv.second[0]);
        patternsToPrint.push_back(std::move(pp));
      }

      // Now print all collected patterns (using their original names)
      for (auto &pp : patternsToPrint) {
        // Use the original helper name that's also in opToHelper
        const std::string &helperName = pp.helperName;

        auto &combinedOps = pp.ops;
        auto &rootOps = pp.rootOps;

        llvm::DenseMap<Operation *, size_t> opToIdx;
        for (size_t i = 0; i < combinedOps.size(); ++i)
          opToIdx[combinedOps[i]] = i;

        // Collect external inputs
        std::vector<Value> extInputs;
        llvm::DenseMap<Value, size_t> inputIdx;
        for (size_t i = combinedOps.size(); i > 0; --i) {
          Operation *op = combinedOps[i - 1];
          for (Value operand : op->getOperands()) {
            Operation *defOp = operand.getDefiningOp();
            if (defOp && opToIdx.count(defOp))
              continue;
            if (!inputIdx.count(operand)) {
              inputIdx[operand] = extInputs.size();
              extInputs.push_back(operand);
            }
          }
        }

        // Topologically sort
        llvm::DenseMap<Operation *, int> inDegree;
        llvm::DenseMap<Operation *, std::vector<Operation *>> dependents;
        for (Operation *op : combinedOps) {
          inDegree[op] = 0;
        }
        for (Operation *op : combinedOps) {
          for (Value operand : op->getOperands()) {
            Operation *defOp = operand.getDefiningOp();
            if (defOp && opToIdx.count(defOp)) {
              inDegree[op]++;
              dependents[defOp].push_back(op);
            }
          }
        }

        std::vector<Operation *> sortedOps;
        std::queue<Operation *> ready;
        for (Operation *op : combinedOps) {
          if (inDegree[op] == 0)
            ready.push(op);
        }
        while (!ready.empty()) {
          Operation *op = ready.front();
          ready.pop();
          sortedOps.push_back(op);
          for (Operation *dep : dependents[op]) {
            if (--inDegree[dep] == 0)
              ready.push(dep);
          }
        }

        // Extract pattern number from helper name for the comment
        size_t patternNum = 0;
        if (helperName.find("_auto_pattern_") == 0) {
          patternNum = std::stoul(helperName.substr(14));
        }

        llvm::outs() << "# Pattern " << patternNum << ": " << combinedOps.size()
                     << " ops, " << pp.numOccurrences << " occurrences";
        if (rootOps.size() > 1) {
          llvm::outs() << ", " << rootOps.size() << " outputs";
        }
        llvm::outs() << "\n";
        llvm::outs() << "def " << helperName << "(";
        for (size_t i = 0; i < extInputs.size(); ++i) {
          if (i > 0)
            llvm::outs() << ", ";
          llvm::outs() << "input" << i;
        }
        llvm::outs() << "):\n";

        llvm::DenseMap<Operation *, std::string> opToTmp;
        size_t tmpCnt = 0;
        for (Operation *op : sortedOps) {
          opToTmp[op] = "tmp" + std::to_string(tmpCnt++);
        }

        for (Operation *op : sortedOps) {
          std::string tmp = opToTmp[op];
          StringRef opName = op->getName().getStringRef();
          StringRef simpleName = opName;
          if (opName.contains("."))
            simpleName = opName.substr(opName.find('.') + 1);

          llvm::outs() << "    " << tmp << " = " << simpleName << "(";
          for (unsigned j = 0; j < op->getNumOperands(); ++j) {
            if (j > 0)
              llvm::outs() << ", ";
            Value operand = op->getOperand(j);
            Operation *defOp = operand.getDefiningOp();
            if (defOp && opToTmp.count(defOp))
              llvm::outs() << opToTmp[defOp];
            else
              llvm::outs() << "input" << inputIdx[operand];
          }
          llvm::outs() << ")\n";
        }

        llvm::outs() << "    return ";
        for (size_t i = 0; i < rootOps.size(); ++i) {
          if (i > 0)
            llvm::outs() << ", ";
          llvm::outs() << opToTmp[rootOps[i]];
        }
        llvm::outs() << "\n\n";
      }

      llvm::outs() << "# ========================================\n\n";
    }

    // Pre-analyze all local_alloc operations
    DenseMap<Operation *, LocalAllocInfo> allocInfoMap;
    m.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "ttg.local_alloc") {
        allocInfoMap[op] = analyzeLocalAlloc(op);
      }
    });

    // Merge opsToSkip into skippedOps
    llvm::DenseSet<Operation *> skippedOps = opsToSkip;

    // Print simplified TLX representation
    for (Region &region : m->getRegions()) {
      printRegion(region, llvm::outs(), opNameMap, allocInfoMap, skippedOps, 0,
                  nullptr, {}, &opToHelper, &opToInputs, &opToCoAnchors);
    }
  }
};

} // namespace tlx
} // namespace triton
} // namespace mlir
