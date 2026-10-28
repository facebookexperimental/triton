#include "test/include/Analysis/TestAxisInfo.h"
#include "third_party/amd/include/Analysis/AxisInfoExt.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"

namespace {

struct AMDTestAxisInfoPass : public mlir::test::TestAxisInfoPass {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AMDTestAxisInfoPass);

  StringRef getArgument() const final { return "test-print-amd-alignment"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<amdgpu::TritonAMDGPUDialect>();
  }

protected:
  ModuleAxisInfoAnalysis getAnalysis(ModuleOp moduleOp) const final {
    return AMD::ModuleAxisInfoAnalysis(moduleOp);
  }
};
} // namespace

namespace mlir::test {
void registerAMDTestAlignmentPass() { PassRegistration<AMDTestAxisInfoPass>(); }
} // namespace mlir::test
