#ifndef TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_DDG_H
#define TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_DDG_H

#include "LatencyModel.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::gpu {

struct DDGEdge {
  unsigned srcIdx{};
  unsigned dstIdx{};
  int latency{};
  unsigned distance{}; // 0 = intra-iteration, 1+ = loop-carried
};

struct DDGNode {
  Operation *op{};
  unsigned idx{};
  HWPipeline pipeline{HWPipeline::NONE};
  int latency{};
  int selfLatency{};
  bool isSuperNode{false}; // True if this node represents an inner loop
  int innerII{0};          // If super-node, the inner loop's II
  int prologueLatency{0};  // If super-node, cycles before TC starts (MEM busy)
  llvm::SmallVector<unsigned> succs;
  llvm::SmallVector<unsigned> preds;
};

/// Data Dependence Graph for one scf.for loop body.
/// Captures both intra-iteration and loop-carried (distance-1) edges.
class DataDependenceGraph {
public:
  static DataDependenceGraph build(scf::ForOp loop, const LatencyModel &model);

  llvm::ArrayRef<DDGNode> getNodes() const { return nodes; }
  llvm::ArrayRef<DDGEdge> getEdges() const { return edges; }
  const DDGNode &getNode(unsigned idx) const { return nodes[idx]; }
  unsigned getNumNodes() const { return nodes.size(); }
  const llvm::DenseMap<Operation *, unsigned> &getOpToIdx() const {
    return opToIdx;
  }

  /// Get all incoming edges for a node.
  llvm::SmallVector<const DDGEdge *> getInEdges(unsigned nodeIdx) const;

  /// Get all outgoing edges for a node.
  llvm::SmallVector<const DDGEdge *> getOutEdges(unsigned nodeIdx) const;

  /// Compute critical-path height (bottom-up) from each node to any sink.
  llvm::DenseMap<unsigned, int> computeCriticalPathHeights() const;

  /// Compute ResMII: max over all pipelines of total self-latency.
  int computeResMII() const;

  /// Compute RecMII: max over all recurrence circuits of sum_lat / sum_dist.
  int computeRecMII() const;

  /// Compute MinII = max(ResMII, RecMII).
  int computeMinII() const;

  /// Dump the DDG to llvm::dbgs() for debugging.
  void dump() const;

private:
  llvm::SmallVector<DDGNode> nodes;
  llvm::SmallVector<DDGEdge> edges;
  llvm::DenseMap<Operation *, unsigned> opToIdx;
  // For multi-stage super-nodes (prologue/kloop/epilogue sharing the same
  // Operation*), opToIdx maps to the epilogue (producer). consumerOpToIdx
  // maps to the prologue so loop-carried edges target the correct node.
  llvm::DenseMap<Operation *, unsigned> consumerOpToIdx;

  unsigned addNode(Operation *op, const LatencyModel &model);
  void addEdge(unsigned src, unsigned dst, int latency, unsigned distance);
};

} // namespace mlir::triton::gpu

#endif // TRITON_NVIDIA_HOPPER_MODULO_SCHEDULING_DDG_H
