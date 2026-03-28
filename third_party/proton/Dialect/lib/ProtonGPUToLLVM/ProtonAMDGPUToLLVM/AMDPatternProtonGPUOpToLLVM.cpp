#include "Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/AMDPatternProtonGPUOpToLLVM.h"
#include "BufferOpsEmitter.h"
#include "Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/TargetInfo.h"
#include "Conversion/ProtonGPUToLLVM/Utility.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct CircularStoreOpConversion
    : public ConvertOpToLLVMPattern<
          mlir::triton::proton::gpu::CircularStoreOp> {
  explicit CircularStoreOpConversion(
      LLVMTypeConverter &typeConverter,
      const proton::gpu::TargetInfoBase &targetInfo, PatternBenefit benefit)
      : mlir::ConvertOpToLLVMPattern<
            mlir::triton::proton::gpu::CircularStoreOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::triton::proton::gpu::CircularStoreOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto dataPack =
        lowerCircularStoreOpHelper(op, adaptor.getSegment(), rewriter);

    uint32_t addrSpace = dataPack.addrSpace;
    if (addrSpace == 1) {
      // Global memory path — use ROCDL buffer operations for branchless
      // predicated stores.  The offset-masking trick avoids branch divergence:
      // when isWriter is false the offset is set past num_records, causing the
      // hardware to silently NOP the store.

      // Build a buffer resource descriptor via BufferEmitter.
      auto &amdTargetInfo = static_cast<const triton::AMD::TargetInfo &>(
          targetInfo.getTritonTargetInfo());
      LLVM::AMD::BufferEmitter bufEmitter(rewriter, loc, amdTargetInfo);
      Value rsrcDesc = bufEmitter.createResourceDescriptor(dataPack.ptr);

      // Branchless predication via offset masking.
      // Real offset = 0 (the pointer already points to the target slot).
      // OOB  offset = 0x80000000 (exceeds num_records → hardware NOP).
      Value realOffset = b.int_val(32, 0);
      Value oobOffset =
          b.int_val(32, static_cast<int32_t>(std::numeric_limits<int>::max() +
                                             int64_t(1)));
      Value maskedOffset = b.select(dataPack.isWriter, realOffset, oobOffset);

      // Scalar offset and cache control.
      // Use SC0=1, NT=1 (cache-streaming / non-temporal) to avoid polluting
      // the L1 data cache with profiling traffic.
      Value sgprOffset = b.int_val(32, 0);
      constexpr int32_t auxCacheStreaming = 0x3; // SC0 | NT
      Value aux = b.int_val(32, auxCacheStreaming);

      // Emit the buffer store.
      // dataPack.record is <2 x i32> (8 bytes: tag+upperClock, lowerClock).
      SmallVector<Value, 5> args{dataPack.record, rsrcDesc, maskedOffset,
                                 sgprOffset, aux};
      rewriter.create<ROCDL::RawPtrBufferStoreOp>(loc, TypeRange{}, args,
                                                  ArrayRef<NamedAttribute>());
    } else if (addrSpace == 3) {
      targetInfo.getTritonTargetInfo().storeDShared(
          rewriter, loc, dataPack.ptr, std::nullopt, dataPack.record,
          dataPack.isWriter);
    } else {
      llvm::report_fatal_error("unsupported address space in circular store");
    }
    rewriter.eraseOp(op);
    return success();
  }

protected:
  const proton::gpu::TargetInfoBase &targetInfo;
};

} // namespace

namespace mlir::triton::proton::gpu::AMD {
void populateProtonGPUOpAMDPatterns(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    const TargetInfo &targetInfo,
                                    PatternBenefit benefit) {
  patterns.add<CircularStoreOpConversion>(typeConverter, targetInfo, benefit);
}
} // namespace mlir::triton::proton::gpu::AMD
