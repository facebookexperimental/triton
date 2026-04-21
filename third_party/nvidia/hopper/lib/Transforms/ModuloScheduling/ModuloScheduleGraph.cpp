// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ModuloScheduleGraph.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::triton::gpu {

static llvm::StringRef memKindName(MemoryKind k) {
  switch (k) {
  case MemoryKind::SMEM:
    return "SMEM";
  case MemoryKind::TMEM:
    return "TMEM";
  case MemoryKind::Register:
    return "Reg";
  case MemoryKind::BARRIER:
    return "BARRIER";
  }
  llvm_unreachable("unknown MemoryKind");
}

static void dumpIndent(llvm::raw_ostream &os, unsigned depth) {
  for (unsigned i = 0; i < depth; ++i)
    os << "  ";
}

static void dumpNodeOneLine(const ScheduleNode &node, llvm::raw_ostream &os,
                            unsigned depth) {
  dumpIndent(os, depth);
  if (node.op && node.op->getNumResults() > 0)
    os << "%N" << node.id << " = ";
  if (node.op)
    os << node.op->getName().getStringRef();
  else
    os << "<null>";
  // Label synthetic inner loop nodes
  if (node.isSuperNode())
    os << " [K-loop]";
  // For ttg.mask: show the first real op inside (1-level unwrap)
  if (node.op && node.op->getNumRegions() > 0 &&
      node.op->getName().getStringRef() == "ttg.mask") {
    auto innerName = [&]() -> llvm::StringRef {
      for (auto &region : node.op->getRegions())
        for (auto &block : region)
          for (auto &inner : block) {
            auto name = inner.getName().getStringRef();
            if (name != "scf.yield" && name != "arith.constant")
              return name;
          }
      return {};
    }();
    if (!innerName.empty())
      os << "(" << innerName << ")";
  }
  os << "  {pipe: " << getPipelineName(node.pipeline)
     << ", cycle: " << node.cycle
     << ", cluster: " << node.cluster;
  if (node.latency)
    os << ", latency: " << node.latency;
  if (node.selfLatency)
    os << ", selfLatency: " << node.selfLatency;
  if (node.warpGroup >= 0)
    os << ", wg: " << node.warpGroup;
  if (node.producesBuffer != UINT_MAX)
    os << ", ->buf" << node.producesBuffer;
  for (auto cb : node.consumesBuffers)
    os << ", <-buf" << cb;
  os << "}\n";
}

static void dumpPort(const ScheduleLoop::MemPort &port, llvm::raw_ostream &os) {
  if (!port.op) {
    if (port.bufferId != UINT_MAX)
      os << "buf" << port.bufferId;
    else
      os << "?";
    return;
  }
  if (port.op->getNumResults() > 0) {
    port.op->getResult(0).printAsOperand(os, OpPrintingFlags());
    os << " : " << port.op->getName().getStringRef();
  } else {
    os << port.op->getName().getStringRef();
  }
}

static void dumpLoop(const ScheduleGraph &graph, const ScheduleLoop &loop,
                     llvm::raw_ostream &os, unsigned depth) {
  dumpIndent(os, depth);
  os << "modulo.schedule @loop" << loop.id << " {\n";
  unsigned inner = depth + 1;

  // Schedule parameters
  dumpIndent(os, inner);
  os << "ii = " << loop.II << ", max_stage = " << loop.maxStage;
  if (loop.prologueLatency)
    os << ", prologue_latency = " << loop.prologueLatency;
  if (loop.tripCount > 0) {
    os << ", trip_count = " << loop.tripCount;
    if (loop.tripCountIsEstimated)
      os << " (estimated)";
  }
  os << "\n";

  // Buffer declarations.
  // Format per design doc §1546-1556:
  //   %buf<id> = modulo.alloc <KIND> [<count> x <shape> x <dtype>]
  //     live=[<start>, <end>)  // <size> bytes total
  //   %bar<id> = modulo.alloc BARRIER [<count>] for buf<paired_id>
  auto dtypeName = [](unsigned bits) -> const char * {
    switch (bits) {
    case 1:
      return "i1";
    case 8:
      return "i8";
    case 16:
      return "f16";
    case 32:
      return "f32";
    case 64:
      return "f64";
    default:
      return "?";
    }
  };
  if (!loop.buffers.empty()) {
    os << "\n";
    for (const auto &buf : loop.buffers) {
      dumpIndent(os, inner);
      if (buf.kind == MemoryKind::BARRIER) {
        os << "%bar" << buf.id << " = modulo.alloc BARRIER [" << buf.count
           << "]";
        if (buf.pairedBufferId != UINT_MAX)
          os << " for buf" << buf.pairedBufferId;
      } else {
        os << "%buf" << buf.id << " = modulo.alloc " << memKindName(buf.kind)
           << " [" << buf.count << " x ";
        for (unsigned i = 0; i < buf.shape.size(); ++i) {
          if (i > 0)
            os << "x";
          os << buf.shape[i];
        }
        os << " x " << dtypeName(buf.elementBitWidth) << "]";
      }
      // Live range (per design doc §215 Step 3 example).
      if (buf.liveStart != 0 || buf.liveEnd != 0)
        os << "  live=[" << buf.liveStart << ", " << buf.liveEnd << ")";
      // Merge group (filled by Step 4.5).
      if (buf.mergeGroupId != UINT_MAX)
        os << "  merged=" << buf.mergeGroupId;
      os << "  // " << buf.sizeBytes() * buf.count << " bytes total\n";
    }

    // Merge groups (per design doc §1555-1556).
    for (const auto &pb : loop.physicalBuffers) {
      dumpIndent(os, inner);
      os << "modulo.merge_group " << pb.id << " {";
      for (size_t i = 0; i < pb.memberBufferIds.size(); ++i) {
        if (i > 0)
          os << ", ";
        os << "buf" << pb.memberBufferIds[i];
      }
      os << "}  // physical: " << pb.sizeBytes << " bytes x " << pb.count
         << "\n";
    }
  }

  // Inputs
  if (!loop.inputs.empty()) {
    os << "\n";
    dumpIndent(os, inner);
    os << "inputs:\n";
    for (const auto &port : loop.inputs) {
      dumpIndent(os, inner + 1);
      dumpPort(port, os);
      os << "\n";
    }
  }

  // Outputs
  if (!loop.outputs.empty()) {
    dumpIndent(os, inner);
    os << "outputs:\n";
    for (const auto &port : loop.outputs) {
      dumpIndent(os, inner + 1);
      dumpPort(port, os);
      os << "\n";
    }
  }

  // Expanded prologue/epilogue (if expanded)
  if (loop.isExpanded) {
    if (!loop.prologueNodes.empty()) {
      os << "\n";
      dumpIndent(os, inner);
      os << "modulo.prologue {\n";
      for (const auto &node : loop.prologueNodes)
        dumpNodeOneLine(node, os, inner + 1);
      dumpIndent(os, inner);
      os << "}\n";
    }
  }

  // Stages (grouped)
  for (int s = 0; s <= loop.maxStage; ++s) {
    auto stageNodes = loop.getNodesInStage(s);
    if (stageNodes.empty())
      continue;
    os << "\n";
    dumpIndent(os, inner);
    os << "modulo.stage @s" << s << " {\n";
    for (const auto *node : stageNodes) {
      if (node->isSuperNode()) {
        dumpLoop(graph, graph.getLoop(node->childPipelineId), os, inner + 1);
      } else {
        dumpNodeOneLine(*node, os, inner + 1);
      }
    }
    dumpIndent(os, inner);
    os << "}\n";
  }

  // Expanded epilogue (if expanded)
  if (loop.isExpanded && !loop.epilogueNodes.empty()) {
    os << "\n";
    dumpIndent(os, inner);
    os << "modulo.epilogue {\n";
    for (const auto &node : loop.epilogueNodes)
      dumpNodeOneLine(node, os, inner + 1);
    dumpIndent(os, inner);
    os << "}\n";
  }

  // Edges
  if (!loop.edges.empty()) {
    os << "\n";
    dumpIndent(os, inner);
    os << "edges {\n";
    for (const auto &edge : loop.edges) {
      dumpIndent(os, inner + 1);
      // Mark super-node endpoints
      auto printNode = [&](unsigned id) {
        os << "N" << id;
        if (id < loop.nodes.size() && loop.nodes[id].isSuperNode())
          os << "(K-loop)";
      };
      printNode(edge.srcId);
      os << " -> ";
      printNode(edge.dstId);
      os << "  lat=" << edge.latency << "  dist=" << edge.distance << "\n";
    }
    dumpIndent(os, inner);
    os << "}\n";
  }

  dumpIndent(os, depth);
  os << "}\n";
}

void ScheduleGraph::dump(llvm::raw_ostream &os) const {
  llvm::DenseSet<unsigned> childIds;
  for (const auto &loop : loops)
    for (const auto &node : loop.nodes)
      if (node.isSuperNode())
        childIds.insert(node.childPipelineId);

  for (const auto &loop : loops) {
    if (childIds.count(loop.id))
      continue;
    dumpLoop(*this, loop, os, 0);
    os << "\n";
  }
}

void ScheduleGraph::dump() const { dump(llvm::dbgs()); }

} // namespace mlir::triton::gpu
