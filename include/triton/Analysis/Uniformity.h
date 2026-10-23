#ifndef TRITON_ANALYSIS_UNIFORMITY_H
#define TRITON_ANALYSIS_UNIFORMITY_H

#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include <functional>
#include <memory>
#include <optional>

namespace mlir::triton {

/// Intra-procedural SIMT uniformity for common MLIR dialects. Uniform means
/// agreement across the participating lanes of a subgroup, not across a CTA.
/// Unknown includes values that may diverge. This is separate from constancy.
///
/// Supports arith, math, index, vector, GPU, LLVM/ROCDL, CF, and sequential SCF
/// control flow. Unmodeled operations/regions are conservative. The input is
/// never modified or lowered. Uses MLIR sparse dataflow with branch and region
/// branch interfaces. CFG joins and cross-block uses conservatively require
/// uniform control throughout their region; no reconvergence or cycle analysis
/// is performed. Recompute after modifying the analyzed function.
class UniformityAnalysis {
public:
  enum class Rule { Uniform, ThreadId, FromOperands, Unknown };
  struct Options {
    /// The X workitem coordinate is the linear thread ID, with subgroup-aligned
    /// boundaries (e.g. a one-dimensional workgroup). Required to seed GPU and
    /// ROCDL X thread IDs. The subgroup size comes from the enclosing module's
    /// ttg.threads-per-warp attribute; missing/invalid sizes disable the seeds.
    bool linearThreadIdX = false;
    /// Function arguments are unknown unless explicitly classified by the
    /// caller. In particular, ordinary device-function arguments are not kernel
    /// arguments. Invoked on the original entry arguments.
    std::function<bool(BlockArgument)> isUniformArgument;
    /// Optional model for ordinary operations and external calls, invoked on
    /// the original operations. Control-flow transfers are modeled separately.
    /// Uniform overrides operand divergence (e.g. subgroup broadcast).
    /// ThreadId seeds the low-bit bound using the subgroup size from the IR.
    std::function<std::optional<Rule>(Operation *, unsigned)> getRule;
  };

  explicit UniformityAnalysis(Operation *function);
  UniformityAnalysis(Operation *function, Options options);
  ~UniformityAnalysis();
  UniformityAnalysis(const UniformityAnalysis &) = delete;
  UniformityAnalysis &operator=(const UniformityAnalysis &) = delete;

  LogicalResult getStatus() const;
  /// Agreement at the definition. Does not imply loop invariance.
  bool isUniform(Value value) const;
  /// Agreement at a particular use, including divergence caused by observing
  /// different iterations after a divergent loop exit.
  bool isUniform(Value value, Operation *user) const;
  bool isUniform(const OpOperand &use) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
} // namespace mlir::triton

#endif
