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
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
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

    // Async store (non-TMA bulk copy)
    {"ttng.async_store", "tlx.async_store",
     "Async store from shared to global memory"},

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
    {"ttng.tensormap_create", "tlx.make_tensor_descriptor",
     "Create TMA descriptor on device (Blackwell)"},
    {"tt.reinterpret_tensor_descriptor", "tlx.reinterpret_tensor_descriptor",
     "Reinterpret TMA descriptor with new shape"},
    {"ttng.reinterpret_tensor_descriptor", "tlx.reinterpret_tensor_descriptor",
     "Reinterpret TMA descriptor (Blackwell)"},
    {"ttg.global_scratch_alloc", "tlx.allocate_tensor_descriptor",
     "Allocate global scratch for TMA descriptors"},
    {"ttng.tensormap_fenceproxy_acquire",
     "tl.extra.cuda.experimental_tensormap_fenceproxy_acquire",
     "Fence proxy acquire for TMA descriptor"},

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
    // Binary arith ops (add, sub, mul, div, rem, xor, and, or) are handled
    // as infix operators (a + b, a * b, etc.) in printSimplifiedOp.
    {"arith.constant", "const", "Constant value"},
    {"arith.select", "select", "Select operation"},
    {"arith.maxf", "tl.maximum", "Float max"},
    {"arith.maxnumf", "tl.maximum", "Float max (NaN-propagating)"},
    {"arith.minf", "tl.minimum", "Float min"},
    {"arith.minnumf", "tl.minimum", "Float min (NaN-propagating)"},
    {"arith.maxsi", "tl.max", "Signed integer max"},
    {"arith.maxui", "tl.max", "Unsigned integer max"},
    {"arith.minsi", "tl.min", "Signed integer min"},
    {"arith.minui", "tl.min", "Unsigned integer min"},

    // Triton operations
    {"tt.splat", "tl.splat", "Splat scalar to tensor"},
    {"tt.broadcast", "tl.broadcast", "Broadcast tensor"},
    {"tt.expand_dims", "tl.expand_dims", "Expand dimensions"},
    {"tt.reduce", "tl.reduce", "Reduce operation"},
    {"tt.dot", "tl.dot", "Matrix multiply"},
    {"tt.load", "tl.load", "Load from global memory"},
    {"tt.store", "tl.store", "Store to global memory"},
    {"tt.addptr", "addptr", "Add to pointer"},
    {"tt.make_range", "tl.arange", "Make range"},
    {"tt.trans", "tl.trans", "Transpose"},
    {"tt.reshape", "tl.reshape", "Reshape tensor"},
    {"tt.cat", "tl.cat", "Concatenate"},
    {"tt.join", "tl.join", "Join tensors"},
    {"tt.split", "tl.split", "Split tensor"},
    {"tt.get_program_id", "tl.program_id", "Get program ID"},
    {"tt.get_num_programs", "tl.num_programs", "Get number of programs"},
    {"tt.return", "return", "Return from function"},

    // Math dialect operations
    {"math.exp", "tl.math.exp", "Natural exponential"},
    {"math.exp2", "tl.math.exp2", "Base-2 exponential"},
    {"math.log", "tl.math.log", "Natural logarithm"},
    {"math.log2", "tl.math.log2", "Base-2 logarithm"},
    {"math.sin", "tl.math.sin", "Sine"},
    {"math.cos", "tl.math.cos", "Cosine"},
    {"math.sqrt", "tl.math.sqrt", "Square root"},
    {"math.rsqrt", "tl.math.rsqrt", "Reciprocal square root"},
    {"math.erf", "tl.math.erf", "Error function"},
    {"math.floor", "tl.math.floor", "Floor"},
    {"math.ceil", "tl.math.ceil", "Ceiling"},
    {"math.fabs", "tl.math.abs", "Float absolute value"},
    {"math.iabs", "tl.math.abs", "Integer absolute value"},
    {"math.fma", "tl.math.fma", "Fused multiply-add"},
    {"tt.precise_sqrt", "tl.math.sqrt_rn", "IEEE-rounded square root"},

    // GPU operations
    {"gpu.barrier", "gpu.barrier", "GPU barrier"},
};

// Infix operator mapping for binary arith ops
llvm::StringMap<StringRef> buildInfixOpMap() {
  llvm::StringMap<StringRef> map;
  map["arith.addi"] = "+";
  map["arith.addf"] = "+";
  map["arith.subi"] = "-";
  map["arith.subf"] = "-";
  map["arith.muli"] = "*";
  map["arith.mulf"] = "*";
  map["arith.divsi"] = "//";
  map["arith.divui"] = "//";
  map["arith.divf"] = "/";
  map["arith.remsi"] = "%";
  map["arith.remui"] = "%";
  map["arith.xori"] = "^";
  map["arith.andi"] = "&";
  map["arith.ori"] = "|";
  map["arith.shli"] = "<<";
  map["arith.shrsi"] = ">>";
  map["arith.shrui"] = ">>";
  return map;
}

// Get comparison operator string for arith.cmpi predicates
StringRef getCmpIOperator(int64_t predicate) {
  switch (predicate) {
  case 0:
    return "=="; // eq
  case 1:
    return "!="; // ne
  case 2:
    return "<"; // slt
  case 3:
    return "<="; // sle
  case 4:
    return ">"; // sgt
  case 5:
    return ">="; // sge
  case 6:
    return "<"; // ult
  case 7:
    return "<="; // ule
  case 8:
    return ">"; // ugt
  case 9:
    return ">="; // uge
  default:
    return "??";
  }
}

// Get comparison operator string for arith.cmpf predicates
StringRef getCmpFOperator(int64_t predicate) {
  switch (predicate) {
  case 0:
    return "False"; // false
  case 1:
    return "=="; // oeq
  case 2:
    return ">"; // ogt
  case 3:
    return ">="; // oge
  case 4:
    return "<"; // olt
  case 5:
    return "<="; // ole
  case 6:
    return "!="; // one
  case 8:
    return "=="; // ueq
  case 9:
    return ">"; // ugt
  case 10:
    return ">="; // uge
  case 11:
    return "<"; // ult
  case 12:
    return "<="; // ule
  case 13:
    return "!="; // une
  case 15:
    return "True"; // true
  default:
    return "??";
  }
}

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

  // Pass through convert_layout and type casts: use the input operand's name
  if (Operation *defOp = v.getDefiningOp()) {
    static const llvm::StringSet<> transparentOps = {
        "ttg.convert_layout", "arith.extui",   "arith.extsi",
        "arith.extf",         "arith.trunci",  "arith.truncf",
        "arith.sitofp",       "arith.uitofp",  "arith.fptosi",
        "arith.fptoui",       "arith.bitcast", "arith.index_cast",
        "arith.index_castui",
    };
    if (transparentOps.contains(defOp->getName().getStringRef()) &&
        defOp->getNumOperands() > 0) {
      return getValueName(defOp->getOperand(0), argSubstitutionMap,
                          inlineConstants);
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
  // - arith cast ops: type coercions transparent in Python
  // - tt.return: function terminator
  // - tt.reduce.return: internal to reduce operation
  static const llvm::StringSet<> opsToSkip = {
      "ttng.init_barrier",  "ttg.warp_return",
      "ttg.warp_yield",     "ttg.warp_specialize.partitions",
      "gpu.barrier",        "arith.constant",
      "ttg.convert_layout", "tt.return",
      "tt.reduce.return",   "arith.extui",
      "arith.extsi",        "arith.extf",
      "arith.trunci",       "arith.truncf",
      "arith.sitofp",       "arith.uitofp",
      "arith.fptosi",       "arith.fptoui",
      "arith.bitcast",      "arith.index_cast",
      "arith.index_castui",
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

static const llvm::StringSet<> castOpsSet = {
    "arith.extui",  "arith.extsi",   "arith.trunci",     "arith.extf",
    "arith.truncf", "arith.bitcast", "arith.sitofp",     "arith.uitofp",
    "arith.fptosi", "arith.fptoui",  "arith.index_cast", "arith.index_castui",
};

static Value resolveThroughCasts(Value v) {
  while (auto *op = v.getDefiningOp()) {
    if (castOpsSet.contains(op->getName().getStringRef()) &&
        op->getNumOperands() > 0)
      v = op->getOperand(0);
    else
      break;
  }
  return v;
}

// Forward declarations
void printRegion(Region &region, llvm::raw_ostream &os,
                 const llvm::StringMap<StringRef> &opNameMap,
                 const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
                 llvm::DenseSet<Operation *> &skippedOps, unsigned indent,
                 DenseMap<Value, Value> *argSubstitutionMap = nullptr,
                 ArrayRef<Value> yieldTargets = {});

struct ForLoopInfo {
  unsigned iterArgIdx; // header block arg index of the iterator
  std::string start;   // init value expression
  std::string end;     // bound expression
  std::string step;    // step expression
  Operation *stepOp;   // addi op to add to skippedOps
};

void printCFRegion(Region &region, llvm::raw_ostream &os,
                   const llvm::StringMap<StringRef> &opNameMap,
                   const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
                   llvm::DenseSet<Operation *> &skippedOps, unsigned indent,
                   DenseMap<Value, Value> *argSubstitutionMap = nullptr);

void printCFBlocks(Block *startBlock, Block *stopBlock, llvm::raw_ostream &os,
                   const llvm::StringMap<StringRef> &opNameMap,
                   const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
                   llvm::DenseSet<Operation *> &skippedOps, unsigned indent,
                   DenseMap<Value, Value> *argSubstitutionMap,
                   llvm::SmallDenseSet<Block *, 16> &visitedBlocks,
                   const DenseMap<Block *, ForLoopInfo> &forLoopHeaders);

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

  // Special handling for tt.reshape - print target shape
  if (opName == "tt.reshape" && op->getNumResults() > 0) {
    if (auto resultType =
            dyn_cast<RankedTensorType>(op->getResult(0).getType())) {
      os << getValueName(op->getResult(0), argSubstitutionMap) << " = ";
      os << "tl.reshape(";
      os << getValueName(op->getOperand(0), argSubstitutionMap) << ", [";
      ArrayRef<int64_t> shape = resultType.getShape();
      for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0)
          os << ", ";
        os << shape[i];
      }
      os << "])\n";
      return;
    }
  }

  // Special handling for binary infix operators (a + b, a * b, etc.)
  {
    static llvm::StringMap<StringRef> infixOpMap = buildInfixOpMap();
    auto infixIt = infixOpMap.find(opName);
    if (infixIt != infixOpMap.end() && op->getNumOperands() == 2 &&
        op->getNumResults() > 0) {
      os << getValueName(op->getResult(0), argSubstitutionMap) << " = "
         << getValueName(op->getOperand(0), argSubstitutionMap) << " "
         << infixIt->second << " "
         << getValueName(op->getOperand(1), argSubstitutionMap) << "\n";
      return;
    }
  }

  // Special handling for unary negation
  if (opName == "arith.negf" && op->getNumOperands() == 1 &&
      op->getNumResults() > 0) {
    os << getValueName(op->getResult(0), argSubstitutionMap) << " = -"
       << getValueName(op->getOperand(0), argSubstitutionMap) << "\n";
    return;
  }

  // Special handling for cmpi/cmpf - print as infix comparison
  if ((opName == "arith.cmpi" || opName == "arith.cmpf") &&
      op->getNumOperands() == 2 && op->getNumResults() > 0) {
    if (auto predAttr = op->getAttrOfType<IntegerAttr>("predicate")) {
      int64_t pred = predAttr.getInt();
      StringRef cmpOp = (opName == "arith.cmpi") ? getCmpIOperator(pred)
                                                 : getCmpFOperator(pred);
      os << getValueName(op->getResult(0), argSubstitutionMap) << " = "
         << getValueName(op->getOperand(0), argSubstitutionMap) << " " << cmpOp
         << " " << getValueName(op->getOperand(1), argSubstitutionMap) << "\n";
      return;
    }
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

// If the condition value is defined by a cmpi/cmpf in the same block as the
// cf.cond_br, return the inlined comparison expression (e.g., "var_0 < var_1")
// and add the defining op to skippedOps so it won't be printed separately.
// Returns empty string if inlining is not possible.
std::string getInlinedCondExpr(Value cond,
                               llvm::DenseSet<Operation *> &skippedOps,
                               DenseMap<Value, Value> *argSubstitutionMap) {
  // Resolve through transparent cast ops to find the actual comparison
  Value resolved = resolveThroughCasts(cond);

  auto *defOp = resolved.getDefiningOp();
  if (!defOp || defOp->getNumOperands() != 2 || defOp->getNumResults() == 0)
    return "";
  auto opName = defOp->getName().getStringRef();
  if (opName != "arith.cmpi" && opName != "arith.cmpf")
    return "";
  auto predAttr = defOp->getAttrOfType<IntegerAttr>("predicate");
  if (!predAttr)
    return "";
  // Only inline if all uses of the comparison result are in CF terminators
  // (cond_br condition or branch operands), which the structured printer
  // handles directly.
  for (auto *user : defOp->getResult(0).getUsers()) {
    if (!user->hasTrait<OpTrait::IsTerminator>())
      return "";
  }
  int64_t pred = predAttr.getInt();
  StringRef cmpOp =
      (opName == "arith.cmpi") ? getCmpIOperator(pred) : getCmpFOperator(pred);
  skippedOps.insert(defOp);
  return getValueName(defOp->getOperand(0), argSubstitutionMap) + " " +
         cmpOp.str() + " " +
         getValueName(defOp->getOperand(1), argSubstitutionMap);
}

// Print non-terminator ops from a block (used by CF-aware printer)
void printBlockOps(Block &block, llvm::raw_ostream &os,
                   const llvm::StringMap<StringRef> &opNameMap,
                   const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
                   llvm::DenseSet<Operation *> &skippedOps, unsigned indent,
                   DenseMap<Value, Value> *argSubstitutionMap) {
  for (Operation &op : block) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      break;
    if (shouldSkipOp(&op, allocInfoMap, skippedOps))
      continue;

    // Reuse the same special-case handling from printBlock
    if (op.getName().getStringRef() == "scf.yield")
      continue;

    if (op.getName().getStringRef() == "ttg.warp_specialize") {
      printWarpSpecialize(&op, os, opNameMap, allocInfoMap, skippedOps, indent);
      continue;
    }
    if (op.getName().getStringRef() == "scf.for") {
      printForOp(&op, os, opNameMap, allocInfoMap, skippedOps, indent,
                 argSubstitutionMap);
      continue;
    }
    if (op.getName().getStringRef() == "scf.if") {
      printIfOp(&op, os, opNameMap, allocInfoMap, skippedOps, indent,
                argSubstitutionMap);
      continue;
    }
    if (op.getNumRegions() > 0) {
      printSimplifiedOp(&op, os, opNameMap, allocInfoMap, indent,
                        argSubstitutionMap);
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

// Print block arg assignments: dest_arg = src_value
// If skipArgIdx >= 0, skip that arg index (used for for-loop iterators).
void printBlockArgAssignments(Block *dest, OperandRange operands,
                              llvm::raw_ostream &os, unsigned indent,
                              DenseMap<Value, Value> *argSubstitutionMap,
                              int skipArgIdx = -1) {
  for (unsigned i = 0; i < dest->getNumArguments() && i < operands.size();
       ++i) {
    if ((int)i == skipArgIdx)
      continue;
    std::string destName =
        getValueName(dest->getArgument(i), argSubstitutionMap);
    std::string srcName = getValueName(operands[i], argSubstitutionMap);
    if (destName != srcName) {
      for (unsigned j = 0; j < indent; ++j)
        os << "  ";
      os << destName << " = " << srcName << "\n";
    }
  }
}

// Detect if a header block represents a for-loop: iter starts at init,
// condition is iter < end, update is iter = iter + step.
bool detectForLoopPattern(Block *header, ForLoopInfo &info,
                          DenseMap<Value, Value> *argSubstitutionMap) {
  if (header->getNumArguments() == 0)
    return false;
  auto condBr = dyn_cast<cf::CondBranchOp>(header->getTerminator());
  if (!condBr)
    return false;

  // Resolve condition through casts to find cmpi
  Value condResolved = resolveThroughCasts(condBr.getCondition());
  auto *cmpiOp = condResolved.getDefiningOp();
  if (!cmpiOp || cmpiOp->getName().getStringRef() != "arith.cmpi")
    return false;
  auto predAttr = cmpiOp->getAttrOfType<IntegerAttr>("predicate");
  if (!predAttr)
    return false;
  int64_t pred = predAttr.getInt();
  // slt (2) or ult (6)
  if (pred != 2 && pred != 6)
    return false;

  // LHS must be a header block arg (the iterator)
  Value lhs = resolveThroughCasts(cmpiOp->getOperand(0));
  auto iterArg = dyn_cast<BlockArgument>(lhs);
  if (!iterArg || iterArg.getOwner() != header)
    return false;
  unsigned iterIdx = iterArg.getArgNumber();

  // Find loop body blocks via BFS from trueDest (not crossing header)
  Block *trueDest = condBr.getTrueDest();
  llvm::SmallDenseSet<Block *, 16> bodyBlocks;
  llvm::SmallVector<Block *, 8> worklist;
  worklist.push_back(trueDest);
  while (!worklist.empty()) {
    Block *b = worklist.pop_back_val();
    if (!b || b == header || bodyBlocks.count(b))
      continue;
    bodyBlocks.insert(b);
    auto *t = b->getTerminator();
    if (auto br = dyn_cast<cf::BranchOp>(t))
      worklist.push_back(br.getDest());
    else if (auto cb = dyn_cast<cf::CondBranchOp>(t)) {
      worklist.push_back(cb.getTrueDest());
      worklist.push_back(cb.getFalseDest());
    }
  }

  // Find step from back-edge predecessor
  Operation *stepOp = nullptr;
  std::string stepStr;
  for (Block *pred : header->getPredecessors()) {
    if (!bodyBlocks.count(pred))
      continue;
    auto *predTerm = pred->getTerminator();
    Value updateVal;
    if (auto br = dyn_cast<cf::BranchOp>(predTerm)) {
      if (br.getDest() == header && iterIdx < br.getDestOperands().size())
        updateVal = br.getDestOperands()[iterIdx];
    } else if (auto cb = dyn_cast<cf::CondBranchOp>(predTerm)) {
      if (cb.getTrueDest() == header &&
          iterIdx < cb.getTrueDestOperands().size())
        updateVal = cb.getTrueDestOperands()[iterIdx];
      else if (cb.getFalseDest() == header &&
               iterIdx < cb.getFalseDestOperands().size())
        updateVal = cb.getFalseDestOperands()[iterIdx];
    }
    if (!updateVal)
      continue;
    Value resolved = resolveThroughCasts(updateVal);
    auto *addOp = resolved.getDefiningOp();
    if (!addOp || addOp->getName().getStringRef() != "arith.addi" ||
        addOp->getNumOperands() != 2)
      return false;
    Value a0 = resolveThroughCasts(addOp->getOperand(0));
    Value a1 = resolveThroughCasts(addOp->getOperand(1));
    if (a0 == iterArg) {
      stepStr = getValueName(a1, argSubstitutionMap);
      stepOp = addOp;
    } else if (a1 == iterArg) {
      stepStr = getValueName(a0, argSubstitutionMap);
      stepOp = addOp;
    } else {
      return false;
    }
    break;
  }
  if (stepStr.empty())
    return false;

  // Find init from non-body predecessor
  std::string initStr;
  for (Block *pred : header->getPredecessors()) {
    if (bodyBlocks.count(pred))
      continue;
    auto *predTerm = pred->getTerminator();
    Value initVal;
    if (auto br = dyn_cast<cf::BranchOp>(predTerm)) {
      if (br.getDest() == header && iterIdx < br.getDestOperands().size())
        initVal = br.getDestOperands()[iterIdx];
    } else if (auto cb = dyn_cast<cf::CondBranchOp>(predTerm)) {
      if (cb.getTrueDest() == header &&
          iterIdx < cb.getTrueDestOperands().size())
        initVal = cb.getTrueDestOperands()[iterIdx];
      else if (cb.getFalseDest() == header &&
               iterIdx < cb.getFalseDestOperands().size())
        initVal = cb.getFalseDestOperands()[iterIdx];
    }
    if (!initVal)
      continue;
    initStr = getValueName(initVal, argSubstitutionMap);
    break;
  }
  if (initStr.empty())
    return false;

  info.iterArgIdx = iterIdx;
  info.start = initStr;
  info.end = getValueName(cmpiOp->getOperand(1), argSubstitutionMap);
  info.step = stepStr;
  info.stepOp = stepOp;
  return true;
}

// Find the immediate post-dominator (merge block) for a cf.cond_br.
// For a simple if-else diamond, this is the single successor shared by
// both branches. We walk forward from each branch to find the first block
// that is reachable from both sides.
Block *findMergeBlock(cf::CondBranchOp condBr) {
  Block *trueDest = condBr.getTrueDest();
  Block *falseDest = condBr.getFalseDest();

  // Simple case: both branches go to the same block
  if (trueDest == falseDest)
    return trueDest;

  // Collect all blocks reachable from trueDest (following unconditional
  // branches only, stopping at conditional branches or blocks with multiple
  // predecessors from outside the chain)
  llvm::SmallDenseSet<Block *, 8> trueReachable;
  Block *b = trueDest;
  while (b) {
    trueReachable.insert(b);
    auto *term = b->getTerminator();
    if (auto br = dyn_cast<cf::BranchOp>(term)) {
      b = br.getDest();
    } else {
      break;
    }
  }

  // Walk from falseDest, find first block also reachable from true side
  b = falseDest;
  while (b) {
    if (trueReachable.count(b))
      return b;
    auto *term = b->getTerminator();
    if (auto br = dyn_cast<cf::BranchOp>(term)) {
      b = br.getDest();
    } else {
      break;
    }
  }

  // No merge found — check if trueDest's successor chain leads to falseDest
  // or vice versa (one-armed if)
  b = trueDest;
  while (b) {
    auto *term = b->getTerminator();
    if (auto br = dyn_cast<cf::BranchOp>(term)) {
      if (br.getDest() == falseDest)
        return falseDest;
      b = br.getDest();
    } else {
      break;
    }
  }
  b = falseDest;
  while (b) {
    auto *term = b->getTerminator();
    if (auto br = dyn_cast<cf::BranchOp>(term)) {
      if (br.getDest() == trueDest)
        return trueDest;
      b = br.getDest();
    } else {
      break;
    }
  }

  return nullptr;
}

// Print a CF region by walking the CFG and emitting structured if/else/while.
// Handles blocks from `startBlock` up to (but not including) `stopBlock`.
void printCFBlocks(Block *startBlock, Block *stopBlock, llvm::raw_ostream &os,
                   const llvm::StringMap<StringRef> &opNameMap,
                   const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
                   llvm::DenseSet<Operation *> &skippedOps, unsigned indent,
                   DenseMap<Value, Value> *argSubstitutionMap,
                   llvm::SmallDenseSet<Block *, 16> &visitedBlocks,
                   const DenseMap<Block *, ForLoopInfo> &forLoopHeaders) {
  Block *current = startBlock;
  while (current && current != stopBlock) {
    if (visitedBlocks.count(current))
      return;
    visitedBlocks.insert(current);

    // Pre-scan: if the block terminates with cf.cond_br whose condition comes
    // from a cmpi/cmpf, mark the comparison as skipped before printing block
    // ops so it gets inlined into the if/while line instead of printed twice.
    std::string preComputedCondExpr;
    if (auto condBrPre = dyn_cast<cf::CondBranchOp>(current->getTerminator())) {
      preComputedCondExpr = getInlinedCondExpr(condBrPre.getCondition(),
                                               skippedOps, argSubstitutionMap);
    }

    // Print non-terminator operations
    printBlockOps(*current, os, opNameMap, allocInfoMap, skippedOps, indent,
                  argSubstitutionMap);

    Operation *term = current->getTerminator();

    // cf.cond_br: emit if/else structure
    if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
      Block *trueDest = condBr.getTrueDest();
      Block *falseDest = condBr.getFalseDest();
      Block *mergeBlock = findMergeBlock(condBr);

      // Check if this is a while loop header: the false branch exits the
      // loop (goes to mergeBlock or stopBlock) and the true branch is the
      // loop body that eventually branches back to current.
      // Pattern: current block has args, true branch leads back to current.
      bool isWhileLoop = false;
      if (current->getNumArguments() > 0) {
        // BFS to check if the true-side eventually branches back to current
        llvm::SmallVector<Block *, 8> worklist;
        llvm::SmallDenseSet<Block *, 16> visited;
        worklist.push_back(trueDest);
        while (!worklist.empty() && !isWhileLoop) {
          Block *walk = worklist.pop_back_val();
          if (!walk || visited.count(walk))
            continue;
          visited.insert(walk);
          if (walk == current) {
            isWhileLoop = true;
            break;
          }
          auto *t = walk->getTerminator();
          if (auto br = dyn_cast<cf::BranchOp>(t)) {
            worklist.push_back(br.getDest());
          } else if (auto cb = dyn_cast<cf::CondBranchOp>(t)) {
            worklist.push_back(cb.getTrueDest());
            worklist.push_back(cb.getFalseDest());
          }
        }
      }

      if (isWhileLoop) {
        // Check if this matches a for-loop pattern
        auto forIt = forLoopHeaders.find(current);
        if (forIt != forLoopHeaders.end()) {
          const ForLoopInfo &fli = forIt->second;
          // Add step op to skippedOps so it's not printed separately
          if (fli.stepOp)
            skippedOps.insert(fli.stepOp);
          int skipIdx = (int)fli.iterArgIdx;

          for (unsigned i = 0; i < indent; ++i)
            os << "  ";
          std::string iterName = getValueName(
              current->getArgument(fli.iterArgIdx), argSubstitutionMap);
          if (fli.step == "1")
            os << "for " << iterName << " in tl.range(" << fli.start << ", "
               << fli.end << "):\n";
          else
            os << "for " << iterName << " in tl.range(" << fli.start << ", "
               << fli.end << ", " << fli.step << "):\n";

          // Print true-dest arg assignments (skip iterator)
          printBlockArgAssignments(trueDest, condBr.getTrueDestOperands(), os,
                                   indent + 1, argSubstitutionMap, skipIdx);

          // Print loop body
          printCFBlocks(trueDest, current, os, opNameMap, allocInfoMap,
                        skippedOps, indent + 1, argSubstitutionMap,
                        visitedBlocks, forLoopHeaders);

          // Continue with exit
          printBlockArgAssignments(falseDest, condBr.getFalseDestOperands(), os,
                                   indent, argSubstitutionMap, skipIdx);
          current = falseDest;
          continue;
        }

        // Regular while loop
        std::string condExpr =
            preComputedCondExpr.empty()
                ? getValueName(condBr.getCondition(), argSubstitutionMap)
                : preComputedCondExpr;
        for (unsigned i = 0; i < indent; ++i)
          os << "  ";
        os << "while " << condExpr << ":\n";

        // Print true-dest arg assignments if any
        printBlockArgAssignments(trueDest, condBr.getTrueDestOperands(), os,
                                 indent + 1, argSubstitutionMap);

        // Print loop body (true branch), stopping when we get back to current
        printCFBlocks(trueDest, current, os, opNameMap, allocInfoMap,
                      skippedOps, indent + 1, argSubstitutionMap, visitedBlocks,
                      forLoopHeaders);

        // After the while, continue with the false dest (exit)
        printBlockArgAssignments(falseDest, condBr.getFalseDestOperands(), os,
                                 indent, argSubstitutionMap);
        current = falseDest;
        continue;
      }

      // Regular if/else
      std::string condExpr =
          preComputedCondExpr.empty()
              ? getValueName(condBr.getCondition(), argSubstitutionMap)
              : preComputedCondExpr;
      for (unsigned i = 0; i < indent; ++i)
        os << "  ";
      os << "if " << condExpr << ":\n";

      // Print true-dest arg assignments
      printBlockArgAssignments(trueDest, condBr.getTrueDestOperands(), os,
                               indent + 1, argSubstitutionMap);

      if (trueDest != mergeBlock) {
        printCFBlocks(trueDest, mergeBlock, os, opNameMap, allocInfoMap,
                      skippedOps, indent + 1, argSubstitutionMap, visitedBlocks,
                      forLoopHeaders);
      }

      // Print else branch if it's not the merge block or has operands
      if (falseDest != mergeBlock || condBr.getFalseDestOperands().size() > 0) {
        for (unsigned i = 0; i < indent; ++i)
          os << "  ";
        os << "else:\n";
        printBlockArgAssignments(falseDest, condBr.getFalseDestOperands(), os,
                                 indent + 1, argSubstitutionMap);
        if (falseDest != mergeBlock) {
          printCFBlocks(falseDest, mergeBlock, os, opNameMap, allocInfoMap,
                        skippedOps, indent + 1, argSubstitutionMap,
                        visitedBlocks, forLoopHeaders);
        }
      }

      // Continue with merge block
      if (mergeBlock) {
        current = mergeBlock;
        continue;
      }
      return;
    }

    // cf.br: unconditional branch — print arg assignments and continue
    if (auto br = dyn_cast<cf::BranchOp>(term)) {
      Block *dest = br.getDest();
      // Skip iterator arg assignment when branching to a for-loop header
      int skipIdx = -1;
      auto forIt = forLoopHeaders.find(dest);
      if (forIt != forLoopHeaders.end())
        skipIdx = (int)forIt->second.iterArgIdx;
      printBlockArgAssignments(dest, br.getDestOperands(), os, indent,
                               argSubstitutionMap, skipIdx);
      // If dest is already visited (back-edge) or is the stop block, stop
      if (visitedBlocks.count(dest) || dest == stopBlock)
        return;
      current = dest;
      continue;
    }

    // Unknown terminator — just stop
    return;
  }
}

// Entry point for CF-aware region printing
void printCFRegion(Region &region, llvm::raw_ostream &os,
                   const llvm::StringMap<StringRef> &opNameMap,
                   const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
                   llvm::DenseSet<Operation *> &skippedOps, unsigned indent,
                   DenseMap<Value, Value> *argSubstitutionMap) {
  if (region.empty())
    return;

  // Pre-scan: detect for-loop headers
  DenseMap<Block *, ForLoopInfo> forLoopHeaders;
  for (Block &block : region) {
    ForLoopInfo info;
    if (detectForLoopPattern(&block, info, argSubstitutionMap))
      forLoopHeaders[&block] = info;
  }

  Block &entry = region.front();
  llvm::SmallDenseSet<Block *, 16> visitedBlocks;
  printCFBlocks(&entry, nullptr, os, opNameMap, allocInfoMap, skippedOps,
                indent, argSubstitutionMap, visitedBlocks, forLoopHeaders);
}

void printRegion(Region &region, llvm::raw_ostream &os,
                 const llvm::StringMap<StringRef> &opNameMap,
                 const DenseMap<Operation *, LocalAllocInfo> &allocInfoMap,
                 llvm::DenseSet<Operation *> &skippedOps, unsigned indent,
                 DenseMap<Value, Value> *argSubstitutionMap,
                 ArrayRef<Value> yieldTargets) {
  // For multi-block regions with CF control flow, use the CF-aware printer
  if (std::distance(region.begin(), region.end()) > 1) {
    printCFRegion(region, os, opNameMap, allocInfoMap, skippedOps, indent,
                  argSubstitutionMap);
    return;
  }
  // Single-block region: print sequentially
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
