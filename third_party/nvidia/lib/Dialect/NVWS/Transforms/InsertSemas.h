#ifndef NVWS_TRANSFORMS_INSERT_SEMAS_H_
#define NVWS_TRANSFORMS_INSERT_SEMAS_H_
#include "lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionAttrs.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace mlir::triton::nvws_semas {
using triton::nvws::AsyncOp;
namespace gpu = triton::gpu;
namespace nvidia_gpu = triton::nvidia_gpu;
namespace nvws = triton::nvws;
using MemberId = unsigned;
using PieceId = unsigned;
using SemaId = unsigned;
using PartitionId = std::pair<int /*ttg.partition*/, int /*ws tag*/>;
using Owner = std::optional<PartitionId>;

inline int64_t ownerKey(const Owner &o) {
  return o ? (static_cast<int64_t>(o->second) << 32) |
                 static_cast<uint32_t>(o->first)
           : -1;
}
inline bool sameOwner(const Owner &a, const Owner &b) { return a == b; }

enum class Effect : uint8_t { R, W };
inline Effect joinEffect(Effect a, Effect b) {
  return (a == Effect::W || b == Effect::W) ? Effect::W : Effect::R;
}

struct PieceInfo {
  Owner owner;
  Effect effect = Effect::R;
};

struct AliasStep {
  Operation *op = nullptr;
  unsigned operandIdx = 0;
  // Snapshot the authored result type before EmitIR remaps operands and
  // refreshes cloned alias result types.
  Type resultType;
};

struct Touch {
  MemberId member = 0;
  Effect effect = Effect::R;
  Value accessValue;
  // Snapshot the authored access type.  The original SSA value may be
  // rewritten while another group is still rendering.
  Type accessType;
  SmallVector<AliasStep, 2> alias;
};

struct Node;

// Static identity of one token.  The group is implicit in the GroupDag being
// processed; producer identifies the token SSA (an acquire or a region
// carrier), while that token preserves its dynamic physical-slot lineage.
// `sema` is the selected render channel; region results preserve an incoming
// channel when one exists, otherwise use a deterministic fallback. The exact
// producer may have acquired another channel in the same GroupDag.
struct TokenRef {
  Node *producer = nullptr;
  SemaId sema = 0;
  Owner owner;
};
// A region that threads a token: each exit path names the boundary owner's
// last token producer inside that path, or nullptr to pass the ENTER token
// through unchanged.
struct RegionFlow {
  Owner owner;
  SmallVector<Node *, 2> exits; // nullptr means pass the ENTER token
  // Render channel of the region's token result.  Placement records only the
  // exact path producers; channel formation fills this after every acquire
  // and release has its final location.
  bool inheritsInputChannel = false;
  std::optional<SemaId> sema;
};

struct Node {
  enum Kind { Func, For, If, Enter, Exit, Access, Acquire, Release };
  static constexpr size_t NumKinds = Release + 1;
  Kind kind = Access;
  Operation *op = nullptr;
  Operation *completionAnchor = nullptr;
  Node *parent = nullptr, *prev = nullptr, *next = nullptr;
  SmallVector<Node *, 2> children;
  Owner owner;
  SmallVector<Touch, 2> touches;
  DenseMap<PieceId, PieceInfo> pieceInfo;
  SemaId sema = 0;
  unsigned count = 0;
  SmallVector<AsyncOp, 1> payloads;
  // Aggregate memory effect for structural slot events. Protocol nodes have
  // no value; Access DAG construction seals Access/region effects once.
  std::optional<Effect> slotEffect;
  gpu::StageCluster stageCluster;
  std::optional<int64_t> stageOffset;
  std::optional<int64_t> bufferStageOffset;
  // Exact distance for a recurrence demand.  In particular, a tail acquire
  // can be lexically after its releases while supplying the next iteration;
  // schedule analysis must not rediscover that fact from list order.
  std::optional<int64_t> recurrenceDistance;
  // Exact token producer for this access or release: the acquire, region, or
  // inherited record whose token this node consumes. Assigned by the SYNC-DAG
  // token sweep; EmitIR routes by this fact and never guesses by owner.
  Node *tokenSource = nullptr;
  Node *releasedViewRelease = nullptr; // Authorizes exact view reuse.
  // Effective owner of an Acquire token or Region token placeholder/result.
  // The outer optional distinguishes "not a producer" from a root-owned token.
  std::optional<Owner> producedTokenOwner;
  Node *sat = nullptr;
  Node *scheduleAnchor = nullptr;
  std::optional<RegionFlow> flow;
  SmallVector<int, 2> requiredParts;
  bool isRegion() const { return kind == For || kind == If; }
};
inline SmallVector<std::pair<PieceId, PieceInfo>, 4>
sortedPieceInfo(const Node *n) {
  SmallVector<std::pair<PieceId, PieceInfo>, 4> v(n->pieceInfo.begin(),
                                                  n->pieceInfo.end());
  llvm::sort(v, [](const auto &a, const auto &b) { return a.first < b.first; });
  return v;
}
// Outer optional: the node has one uniform owner. Inner Owner may still be
// root; an empty outer optional means no pieces or mixed owners.
inline std::optional<Owner> uniformPieceOwner(const Node *n) {
  if (n->pieceInfo.empty())
    return std::nullopt;
  Owner owner = n->pieceInfo.begin()->second.owner;
  for (const auto &entry : n->pieceInfo)
    if (!sameOwner(owner, entry.second.owner))
      return std::nullopt;
  return std::optional<Owner>(std::in_place, owner);
}

struct Member {
  Operation *allocOp = nullptr;
  gpu::MemDescType type;
  int64_t offset = 0;
  int64_t extent = 1;
  int64_t copies = 1;
  int64_t circularStart = 0;
  MemberId backingPrimary = 0;
};

struct PieceTable {
  SmallVector<Member> members;
  SmallVector<SmallVector<PieceId, 2>> footprint;
};

struct Sema {
  std::string name;
  unsigned count = 0;
  std::optional<Owner> entryOwner;
  std::optional<uint32_t> releasedMask;
  const Sema *physicalPrimary = nullptr;
};

enum class MemKind { Tmem, Local };

struct GroupDag {
  int64_t bufferId = 0;
  MemKind memory = MemKind::Tmem;
  bool circular = false;
  PieceTable pieceTable;
  DenseMap<Value, std::pair<MemberId, SmallVector<AliasStep, 2>>> aliases;
  SmallVector<Operation *, 1> ttDescriptorFedMembers;
  DenseSet<Operation *> accessNodeOps;
  SmallVector<std::unique_ptr<Node>> nodes;
  Node *root = nullptr;
  SmallVector<Sema> semas;
  int numCopies = 1, numSemaphoreCopies = 1;
  GroupDag *physicalBacking = nullptr;
  Operation *semaAnchor = nullptr, *physicalBackingAnchor = nullptr;
  bool isTmem() const { return memory == MemKind::Tmem; }
  bool isCircular() const { return circular; }
  ArrayRef<Node *> nodesOfKind(Node::Kind kind) const {
    return nodesByKind[kind];
  }
  bool isSealed() const { return sealed; }
  void seal() {
    assert(!sealed);
    sealed = true;
  }
  Node *newNode(Node::Kind k, Operation *op, Node *parent) {
    assert(!sealed && "cannot mutate a sealed SYNC-DAG");
    Node *n = nodes.emplace_back(std::make_unique<Node>()).get();
    n->kind = k;
    n->op = op;
    n->parent = parent;
    nodesByKind[k].push_back(n);
    return n;
  }
  void discardLastNode(Node *node) {
    assert(!sealed && "cannot mutate a sealed SYNC-DAG");
    assert(nodes.back().get() == node &&
           "only the newest node can be discarded");
    assert(nodesByKind[node->kind].back() == node &&
           "kind index must follow allocation order");
    nodesByKind[node->kind].pop_back();
    nodes.pop_back();
  }

private:
  bool sealed = false;
  // Allocation-order index. Only newNode/discardLastNode may mutate it.
  std::array<SmallVector<Node *, 0>, Node::NumKinds> nodesByKind;
};
inline gpu::MemDescType backingType(const GroupDag &g, const Member &member) {
  auto type = member.type;
  SmallVector<int64_t> shape(type.getShape());
  if (!isa<nvidia_gpu::TensorMemoryScalesEncodingAttr>(type.getEncoding()))
    shape.insert(shape.begin(), g.numCopies);
  return gpu::MemDescType::get(shape, type.getElementType(), type.getEncoding(),
                               type.getMemorySpace(), true);
}
using ScheduleUpdate = std::pair<Operation *, gpu::StageCluster>;
inline constexpr StringLiteral kBufferIdAttrName = "buffer.id";
inline constexpr StringLiteral kBufferOffsetAttrName = "buffer.offset";
inline constexpr StringLiteral kBufferCopyAttrName = "buffer.copy";
inline constexpr StringLiteral kBufferCircularAttrName = "buffer.circular";
inline constexpr StringLiteral kBufferStartAttrName = "buffer.start";
inline std::optional<int64_t> getI64Attr(Operation *op, StringRef name) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return std::nullopt;
}
inline InFlightDiagnostic semaError(Operation *op) {
  return op->emitError() << "nvws-insert-semas: ";
}
template <typename OpTy> inline InFlightDiagnostic semaError(OpTy op) {
  return semaError(op.getOperation());
}
inline Owner resolveOwner(Operation *op) {
  if (!gpu::hasPartition(op))
    return std::nullopt;
  auto ids = gpu::getPartitionIds(op);
  if (ids.size() != 1)
    return std::nullopt;
  Operation *tagSrc = op;
  while (tagSrc && !gpu::hasWarpSpecializeTag(tagSrc))
    tagSrc = tagSrc->getParentOfType<scf::ForOp>();
  if (!tagSrc)
    return std::nullopt;
  return PartitionId{*ids.begin(), *gpu::getWarpSpecializeTag(tagSrc)};
}
inline std::string ownerStr(Operation *anchor, const Owner &owner) {
  if (!owner)
    return "root";
  std::string s;
  llvm::raw_string_ostream os(s);
  Operation *scope = anchor;
  while (scope && !(isa<scf::ForOp>(scope) && gpu::hasWarpSpecializeTag(scope)))
    scope = scope->getParentOfType<scf::ForOp>();
  std::optional<int> tag =
      scope ? gpu::getWarpSpecializeTag(scope) : std::nullopt;
  if (tag && *tag == owner->second)
    os << "{" << owner->first << "}";
  else
    os << "{@" << owner->second << "." << owner->first << "}";
  return s;
}
inline AsyncOp asyncPayloadOf(Operation *op) {
  if (!op)
    return AsyncOp::NONE;
  if (auto localAlloc = dyn_cast<gpu::LocalAllocOp>(op))
    if (Value src = localAlloc.getSrc())
      if (Operation *def = src.getDefiningOp())
        return asyncPayloadOf(def);
  if (isa<nvidia_gpu::MMAv5OpInterface>(op))
    return AsyncOp::TC5MMA;
  StringRef name = op->getName().getStringRef();
  if (name == "nvws.descriptor_load" || name == "nvws.descriptor_gather")
    return AsyncOp::TMALoad;
  return AsyncOp::NONE;
}
inline bool isSupportedAliasOp(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "ttg.memdesc_index" || name == "ttg.memdesc_subview" ||
         name == "ttg.memdesc_subslice" || name == "ttg.memdesc_trans" ||
         name == "ttg.memdesc_reinterpret" || name == "ttg.memdesc_reshape";
}
template <typename Fn> inline void forEachNode(Node *head, Fn &&fn) {
  for (Node *n = head; n; n = n->next) {
    fn(n);
    for (Node *child : n->children)
      forEachNode(child, fn);
  }
}
template <typename Fn> inline void forEachNode(GroupDag &g, Fn &&fn) {
  for (Node *child : g.root->children)
    forEachNode(child, fn);
}
template <typename Fn> inline void forEachRegionPostOrder(Node *head, Fn &&fn) {
  for (Node *n = head; n; n = n->next) {
    if (!n->isRegion())
      continue;
    for (Node *child : n->children)
      forEachRegionPostOrder(child, fn);
    fn(n);
  }
}
template <typename Fn>
inline void forEachTouchedPiece(const GroupDag &g, const Node *node, Fn &&fn) {
  for (const Touch &touch : node->touches)
    for (PieceId piece : g.pieceTable.footprint[touch.member])
      fn(piece, touch.effect);
}
inline bool touchesPiece(const GroupDag &g, const Node *node, PieceId piece) {
  bool found = false;
  forEachTouchedPiece(g, node, [&](PieceId p, Effect) { found |= p == piece; });
  return found;
}
template <typename Map>
inline void mergeEffect(Map &effects, PieceId piece, Effect effect) {
  effects[piece] = joinEffect(effects[piece], effect);
}
bool shouldDumpDag();
void dumpSyncDagTree(GroupDag &g);
void dumpSyncDagTrees(MutableArrayRef<GroupDag> groups);
void dumpSyncDags(MutableArrayRef<GroupDag> groups, triton::FuncOp func);
FailureOr<SmallVector<GroupDag, 0>>
collectGroups(triton::FuncOp funcOp, Block *functionBlock = nullptr);
LogicalResult buildAccessDag(GroupDag &g, triton::FuncOp funcOp,
                             Block &functionBlock);
LogicalResult buildSyncDag(GroupDag &g, bool useMetaPartitioner,
                           int lowerSemaphoreNumStages, int &numTmemBlocks);
LogicalResult finalizeSyncSchedule(MutableArrayRef<GroupDag> groups,
                                   SmallVectorImpl<ScheduleUpdate> &updates);
LogicalResult emitIR(triton::FuncOp funcOp, ArrayRef<GroupDag> groups,
                     ArrayRef<ScheduleUpdate> updates);
} // namespace mlir::triton::nvws_semas
#endif // NVWS_TRANSFORMS_INSERT_SEMAS_H_
