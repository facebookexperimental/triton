#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "mlir/Pass/PassManager.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/Utility/CommonUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"



namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;
namespace ttng = mlir::triton::nvidia_gpu;

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-lower-barrier-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPULOWERBARRIEROPS
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

static const int THREADS_PER_WAVE = 64;
static const int WAVES_PER_TASK = 4;

void lowerArriveBarrierOps(ModuleOp m, const std::map<mlir::triton::gpu::LocalAllocOp, int> &localAllocToBarrierExpectedCount) {
  SmallVector<Operation *> eraseOps;
  auto cond = Value();
  m.walk([&](ttng::ArriveBarrierOp op) {
    LDBG("Lowering ArriveBarrierOp: " << op << "\n");
    auto loc = op.getLoc();
    OpBuilder builder(op);
    if (!cond) {
      auto ctx = op.getContext();
      // Create if condition for the arrive
      auto i32ty = builder.getIntegerType(32);
      auto threadId = builder.create<ROCDL::ThreadIdXOp>(loc, i32ty);
      auto threadsPerWave =
          builder.create<arith::ConstantIntOp>(loc, THREADS_PER_WAVE, 32);
      auto mod = builder.create<arith::RemSIOp>(loc, threadId, threadsPerWave);
      auto zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
      cond =
          builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, mod, zero);
    }
    auto ifOp = builder.create<scf::IfOp>(loc, cond);
    auto thenBuilder = ifOp.getThenBodyBuilder();
    if (auto defOp = dyn_cast<triton::gpu::MemDescIndexOp>(op.getAlloc().getDefiningOp())) {
      if (auto srcOp = dyn_cast<triton::gpu::LocalAllocOp>(defOp.getSrc().getDefiningOp())) {
        auto it = localAllocToBarrierExpectedCount.find(srcOp);
        if (it != localAllocToBarrierExpectedCount.end()) {
          auto expectedCount = it->second;
          auto incrementCount = op.getCount();
          LDBG("srcOp: " << srcOp << " inc: " << incrementCount << "expected: " << expectedCount << "\n");
          thenBuilder.create<triton::amdgpu::ArriveBarrierOp>(loc, op.getAlloc(), incrementCount, expectedCount);
        } else {
          assert(false && "Cannot find LocalAlllocOp for ArriveBarrierOp");
        }
      } 
    } else {
      assert(false && "ArriveBarrierOp not connected to LocalAllocOp");
    }
    eraseOps.push_back(op);
  });
  for (auto op : eraseOps) { op->erase(); }
}

void lowerWaitBarrierOps(ModuleOp m) {
  SmallVector<Operation *> eraseOps;
  m.walk([&](ttng::WaitBarrierOp op) {
    LDBG("Lowering WaitBarrierOp: " << op << "\n");
    auto loc = op.getLoc();
    OpBuilder builder(op);
    auto waitPhase = op.getPhase();
    auto whileOp = builder.create<scf::WhileOp>(loc, TypeRange{}, ValueRange{});
    // Spin Wait
    // while - Before block
    Block *beforeBlock = builder.createBlock(&whileOp.getBefore());
    builder.setInsertionPointToEnd(beforeBlock);
    auto i32ty = builder.getIntegerType(32);
    // TODO: Lower this to a LocalLoad
    Value barrierPhase = builder.create<triton::amdgpu::ReadBarrierPhaseOp>(loc, i32ty, op.getAlloc());
    Value phaseCond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  barrierPhase, waitPhase);
    builder.create<scf::ConditionOp>(loc, phaseCond, ValueRange{});
     // while - after block
    Block *afterBlock = builder.createBlock(&whileOp.getAfter());
    builder.setInsertionPointToEnd(afterBlock);
    auto five = builder.create<arith::ConstantIntOp>(loc, 5, 32);
    auto sleepInstrinsic = "llvm.amdgcn.s.sleep";
    auto sleepOp = LLVM::createLLVMIntrinsicCallOp(builder, loc, sleepInstrinsic, TypeRange{}, ValueRange{five});
    builder.create<scf::YieldOp>(loc,  ValueRange{});
    builder.setInsertionPointAfter(whileOp);
  
    const char *asmStr = "s_wakeup";
    const char *constraints = "";
    auto asmDialectAttr = LLVM::AsmDialectAttr::get(builder.getContext(),
                                                      LLVM::AsmDialect::AD_ATT);
    builder.create<LLVM::InlineAsmOp>(
        loc,
        /*resultTypes=*/TypeRange(), /*operands=*/ValueRange(),
        /*asm_string=*/asmStr, constraints, /*has_side_effects=*/true,
        /*is_align_stack=*/false, LLVM::TailCallKind::None,
        /*asm_dialect=*/asmDialectAttr,
        /*operand_attrs=*/ArrayAttr());// end spin wait

    eraseOps.push_back(op);
  });
  for (auto op : eraseOps) { op->erase();}
}

void lowerInitBarrierOps(ModuleOp m, std::map<mlir::triton::gpu::LocalAllocOp, int> &localAllocToBarrierExpectedCount) {
  SmallVector<Operation *> eraseOps;
  auto cond = Value();
  m.walk([&](ttng::InitBarrierOp op) {
    LDBG("Lowering InitBarrierOp: " << op << "\n");
    auto loc = op.getLoc();
    OpBuilder builder(op);
    if (!cond) {
      auto ctx = op.getContext();
      // Create if tid == 0 condition for the arrive
      auto i32ty = builder.getIntegerType(32);
      auto threadId = builder.create<ROCDL::ThreadIdXOp>(loc, i32ty);
      auto zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
      cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               threadId, zero);
    }
    auto ifOp = builder.create<scf::IfOp>(loc, cond);
    auto thenBuilder = ifOp.getThenBodyBuilder();
    thenBuilder.create<triton::amdgpu::InitBarrierOp>(loc, op.getAlloc(), op.getCount());
    if (auto defOp = dyn_cast<triton::gpu::MemDescIndexOp>(op.getAlloc().getDefiningOp())) {
      if (auto srcOp = dyn_cast<triton::gpu::LocalAllocOp>(defOp.getSrc().getDefiningOp())) {
        LDBG("srcOp: " << srcOp << " count: " << op.getCount() << "\n");
        localAllocToBarrierExpectedCount[srcOp] = op.getCount();
      } 
    } else {
      assert(false && "InitBarrierOp not connected to LocalAllocOp");
    }
    eraseOps.push_back(op);
  });
  for (auto op : eraseOps) { op->erase(); }
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct TritonAMDGPULowerBarrierOpsPass
    : public impl::TritonAMDGPULowerBarrierOpsBase<
          TritonAMDGPULowerBarrierOpsPass> {
public:
  using impl::TritonAMDGPULowerBarrierOpsBase<
      TritonAMDGPULowerBarrierOpsPass>::TritonAMDGPULowerBarrierOpsBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    // Barrier arrive needs expected_arrival_count so it can be reset at phase flip
    // This map saves the expected_arrival_count while processing InitBarrierOp
    std::map<mlir::triton::gpu::LocalAllocOp, int> localAllocToBarrierExpectedCount;
    lowerInitBarrierOps(m, localAllocToBarrierExpectedCount);
    LLVM_DEBUG({
      for (auto [op, count] : localAllocToBarrierExpectedCount) {
        llvm::dbgs() << "localAllocToBarrierExpectedCount " << op << " " << count << "\n";
      }
    });
    lowerArriveBarrierOps(m, localAllocToBarrierExpectedCount);
    lowerWaitBarrierOps(m);  
  }
};

} // namespace mlir
