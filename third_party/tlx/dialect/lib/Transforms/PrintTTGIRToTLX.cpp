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
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

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
             const DenseMap<Value, Value> *argSubstitutionMap = nullptr,
             bool inlineConstants = true) {
  // Check if this value should be substituted
  if (argSubstitutionMap) {
    auto it = argSubstitutionMap->find(v);
    if (it != argSubstitutionMap->end()) {
      // Recursively get the name of the substituted value (without
      // substitution)
      return getValueName(it->second, nullptr, inlineConstants);
    }
  }

  // Inline constants: if this value is defined by arith.constant, return the
  // literal value
  if (inlineConstants) {
    if (Operation *defOp = v.getDefiningOp()) {
      if (defOp->getName().getStringRef() == "arith.constant") {
        if (auto valueAttr = defOp->getAttr("value")) {
          std::string result;
          llvm::raw_string_ostream os(result);
          if (auto intAttr = dyn_cast<IntegerAttr>(valueAttr)) {
            if (intAttr.getType().isInteger(1)) {
              os << (intAttr.getValue().getBoolValue() ? "True" : "False");
            } else {
              os << intAttr.getValue();
            }
          } else if (auto floatAttr = dyn_cast<FloatAttr>(valueAttr)) {
            SmallString<16> str;
            floatAttr.getValue().toString(str);
            os << str;
          } else {
            // Fall through to normal name handling for unsupported constant
            // types
            goto normal_name;
          }
          os.flush();
          return result;
        }
      }
    }
  }

normal_name:
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
// a barrier alloc or not meaningful in TLX output
bool shouldSkipOp(Operation *op,
                  const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
                  llvm::DenseSet<Operation *> &skippedOps) {
  StringRef opName = op->getName().getStringRef();

  // Operations to skip in TLX output:
  // - ttng.init_barrier: folded into alloc_barriers
  // - ttg.warp_return/warp_yield: implicit in with block structure
  // - ttg.warp_specialize.partitions: not meaningful in TLX format
  // - gpu.barrier: not needed in TLX
  // - arith.constant: values are inlined at use sites
  // - ttg.convert_layout: internal layout conversion
  // - tt.return: function terminator
  // - tt.reduce.return: internal to reduce operation
  static const llvm::StringSet<> opsToSkip = {
      "ttng.init_barrier",  "ttg.warp_return",
      "ttg.warp_yield",     "ttg.warp_specialize.partitions",
      "gpu.barrier",        "arith.constant",
      "ttg.convert_layout", "tt.return",
      "tt.reduce.return",
  };
  if (opsToSkip.contains(opName)) {
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
void printRegion(Region &region, llvm::raw_ostream &os,
                 const llvm::StringMap<StringRef> &opNameMap,
                 const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
                 llvm::DenseSet<Operation *> &skippedOps, unsigned indent,
                 DenseMap<Value, Value> *argSubstitutionMap = nullptr,
                 ArrayRef<Value> yieldTargets = {});

// Print scf.for in Python range syntax
void printForOp(Operation *op, llvm::raw_ostream &os,
                const llvm::StringMap<StringRef> &opNameMap,
                const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
                llvm::DenseSet<Operation *> &skippedOps, unsigned indent,
                DenseMap<Value, Value> *argSubstitutionMap = nullptr);

// Print scf.if with yield-to-assignment conversion
void printIfOp(Operation *op, llvm::raw_ostream &os,
               const llvm::StringMap<StringRef> &opNameMap,
               const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
               llvm::DenseSet<Operation *> &skippedOps, unsigned indent,
               DenseMap<Value, Value> *argSubstitutionMap = nullptr);

// Print scf.for in Python range syntax
void printForOp(Operation *op, llvm::raw_ostream &os,
                const llvm::StringMap<StringRef> &opNameMap,
                const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
                llvm::DenseSet<Operation *> &skippedOps, unsigned indent,
                DenseMap<Value, Value> *argSubstitutionMap) {
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
              argSubstitutionMap);
}

// Print scf.if with yield-to-assignment conversion
void printIfOp(Operation *op, llvm::raw_ostream &os,
               const llvm::StringMap<StringRef> &opNameMap,
               const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
               llvm::DenseSet<Operation *> &skippedOps, unsigned indent,
               DenseMap<Value, Value> *argSubstitutionMap) {
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
                indent + 1, argSubstitutionMap, ifResults);
  }

  // Print else region if it exists and is non-empty
  if (op->getNumRegions() > 1 && !op->getRegion(1).empty()) {
    for (unsigned i = 0; i < indent; ++i)
      os << "  ";
    os << "else:\n";
    printRegion(op->getRegion(1), os, opNameMap, allocInfoMap, skippedOps,
                indent + 1, argSubstitutionMap, ifResults);
  }
}

// Helper to check if a region has meaningful operations (not just skipped ops)
bool regionHasMeaningfulOps(
    Region &region, const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
    llvm::DenseSet<Operation *> &skippedOps) {
  for (Block &block : region) {
    for (Operation &op : block) {
      // Skip operations that would be filtered out
      if (shouldSkipOp(&op, allocInfoMap, skippedOps))
        continue;
      // Skip scf.yield as it's handled specially
      if (op.getName().getStringRef() == "scf.yield")
        continue;
      // Found a meaningful operation
      return true;
    }
  }
  return false;
}

// Print warp_specialize operation in TLX async_tasks format
void printWarpSpecialize(
    Operation *op, llvm::raw_ostream &os,
    const llvm::StringMap<StringRef> &opNameMap,
    const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
    llvm::DenseSet<Operation *> &skippedOps, unsigned indent) {
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
                  &argSubstitutionMap);
    } else {
      // Subsequent regions contain ttg.warp_specialize.partitions
      // which has multiple regions (one per partition)
      for (Block &block : region) {
        for (Operation &innerOp : block) {
          if (innerOp.getName().getStringRef() ==
              "ttg.warp_specialize.partitions") {
            // Each region in warp_specialize.partitions is a partition
            for (Region &partitionRegion : innerOp.getRegions()) {
              // Skip empty partitions (only contain skipped ops)
              if (!regionHasMeaningfulOps(partitionRegion, allocInfoMap,
                                          skippedOps)) {
                continue;
              }

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
                          skippedOps, indent + 2, &argSubstitutionMap);
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
    const DenseMap<Value, Value> *argSubstitutionMap = nullptr) {
  StringRef opName = op->getName().getStringRef();

  // Print indentation
  for (unsigned i = 0; i < indent; ++i)
    os << "  ";

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
void printBlock(Block &block, llvm::raw_ostream &os,
                const llvm::StringMap<StringRef> &opNameMap,
                const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
                llvm::DenseSet<Operation *> &skippedOps, unsigned indent,
                DenseMap<Value, Value> *argSubstitutionMap = nullptr,
                ArrayRef<Value> yieldTargets = {}) {
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
                    argSubstitutionMap);
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
                    argSubstitutionMap);
      }
      os << "}\n";
      continue;
    }

    // Check if we should skip this operation
    if (shouldSkipOp(&op, allocInfoMap, skippedOps)) {
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
      printWarpSpecialize(&op, os, opNameMap, allocInfoMap, skippedOps, indent);
      continue;
    }

    // Special handling for scf.for - Python range syntax
    if (op.getName().getStringRef() == "scf.for") {
      printForOp(&op, os, opNameMap, allocInfoMap, skippedOps, indent,
                 argSubstitutionMap);
      continue;
    }

    // Special handling for scf.if - Python if/else with yield-to-assignment
    if (op.getName().getStringRef() == "scf.if") {
      printIfOp(&op, os, opNameMap, allocInfoMap, skippedOps, indent,
                argSubstitutionMap);
      continue;
    }

    // Handle operations with regions (while, etc.)
    if (op.getNumRegions() > 0) {
      printSimplifiedOp(&op, os, opNameMap, allocInfoMap, indent,
                        argSubstitutionMap);
      // Print indentation and opening brace
      for (unsigned i = 0; i < indent; ++i)
        os << "  ";
      os << "{\n";
      for (Region &region : op.getRegions()) {
        printRegion(region, os, opNameMap, allocInfoMap, skippedOps, indent + 1,
                    argSubstitutionMap);
      }
      for (unsigned i = 0; i < indent; ++i)
        os << "  ";
      os << "}\n";
    } else {
      printSimplifiedOp(&op, os, opNameMap, allocInfoMap, indent,
                        argSubstitutionMap);
    }
  }
}

void printRegion(Region &region, llvm::raw_ostream &os,
                 const llvm::StringMap<StringRef> &opNameMap,
                 const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
                 llvm::DenseSet<Operation *> &skippedOps, unsigned indent,
                 DenseMap<Value, Value> *argSubstitutionMap,
                 ArrayRef<Value> yieldTargets) {
  for (Block &block : region) {
    printBlock(block, os, opNameMap, allocInfoMap, skippedOps, indent,
               argSubstitutionMap, yieldTargets);
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

    // Pre-analyze all local_alloc operations
    DenseMap<Operation *, LocalAllocInfo> allocInfoMap;
    m.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "ttg.local_alloc") {
        allocInfoMap[op] = analyzeLocalAlloc(op);
      }
    });

    // Track ops to skip
    llvm::DenseSet<Operation *> skippedOps;

    // Print simplified TLX representation
    for (Region &region : m->getRegions()) {
      printRegion(region, llvm::outs(), opNameMap, allocInfoMap, skippedOps, 0);
    }
  }
};

} // namespace tlx
} // namespace triton
} // namespace mlir
