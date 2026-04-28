#include "InsertSemas.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/ADT/BitVector.h"

namespace mlir::triton::nvws_semas {

namespace {
class SyncDagDumper {
public:
  explicit SyncDagDumper(GroupDag &group) : g(group) {}
  void printTree() {
    for (Node *head : g.root->children)
      printChain(head, 1, nullptr, 0);
  }

  void print(triton::FuncOp func) {
    os << "SYNC-DAG\n|- func @" << func.getName() << "\n";
    printTree();
    if (g.semas.empty()) {
      os << "  BACKING: untouched (no semaphores)\n";
      return;
    }
    os << "  SEMAS: ";
    llvm::interleave(
        g.semas, os,
        [&](const Sema &sema) {
          os << sema.name << "{count=" << sema.count;
          if (sema.entryOwner)
            os << " entry inherit=" << ownerStr(nullptr, *sema.entryOwner);
          os << "}";
        },
        " ");
    os << "\n  BACKING: numCopies=" << g.numCopies << "\n";
  }

private:
  void printPrefix(unsigned depth) {
    while (depth--)
      os << "|  ";
  }
  StringRef semaName(const Node *node) const {
    return node->sema < g.semas.size() ? StringRef(g.semas[node->sema].name)
                                       : StringRef("<unformed>");
  }

  void printPieces(const Node *node, Operation *anchor) {
    if (node->pieceInfo.empty())
      return;
    os << " pieces{";
    llvm::interleaveComma(sortedPieceInfo(node), os, [&](const auto &entry) {
      os << "P" << entry.first << ":"
         << (entry.second.effect == Effect::W ? "W" : "R") << ":"
         << ownerStr(anchor, entry.second.owner);
    });
    os << "}";
  }

  void printYield(const Node *region, unsigned index) {
    if (!region || !region->flow)
      return;
    Node *final = index < region->flow->exits.size()
                      ? region->flow->exits[index]
                      : nullptr;
    os << " yield{";
    if (!final)
      os << "pass";
    else if (final->kind == Node::Acquire || final->kind == Node::Release)
      os << (final->kind == Node::Acquire ? "a " : "r ") << semaName(final);
    else if (final->isRegion())
      os << (final->kind == Node::For ? "scf.for" : "scf.if");
    else
      llvm_unreachable("invalid region result");
    os << "}";
  }

  void printRegion(const Node *node, Operation *anchor) {
    printPieces(node, anchor);
    if (!node->requiredParts.empty()) {
      os << " parts{";
      llvm::interleaveComma(node->requiredParts, os);
      os << "}";
    }
    if (node->flow)
      os << " thread{" << ownerStr(node->op, node->flow->owner) << "}";
    os << "\n";
  }

  void printChain(const Node *head, unsigned depth, const Node *parent,
                  unsigned index) {
    for (const Node *node = head; node; node = node->next) {
      Operation *anchor = node->parent ? node->parent->op : nullptr;
      printPrefix(depth);
      switch (node->kind) {
      case Node::Access:
        os << "|- ";
        llvm::interleaveComma(node->touches, os, [&](const Touch &touch) {
          os << (touch.effect == Effect::W ? "W" : "R") << " m" << touch.member;
        });
        os << "  " << node->op->getName().getStringRef() << " "
           << ownerStr(node->op, node->owner) << "\n";
        break;
      case Node::Acquire:
      case Node::Release:
        os << "|- " << (node->kind == Node::Acquire ? "a" : "r") << "  "
           << semaName(node);
        if (node->count > 1)
          os << "(" << node->count << ")";
        os << "  " << ownerStr(anchor, node->owner);
        if (node->kind == Node::Acquire) {
          if (node->sema < g.semas.size() && g.semas[node->sema].entryOwner &&
              !node->owner)
            os << "  ; entry";
        } else {
          os << " [";
          llvm::interleaveComma(node->payloads, os, [&](AsyncOp payload) {
            os << nvws::stringifyAsyncOp(payload);
          });
          os << "]";
        }
        if (node->stageOffset)
          os << "  stage-offset=" << *node->stageOffset;
        os << "\n";
        break;
      case Node::For:
        os << "|- scf.for";
        if (gpu::hasWarpSpecializeTag(node->op))
          os << " (WS, tag=" << *gpu::getWarpSpecializeTag(node->op) << ")";
        printRegion(node, node->op);
        printChain(node->children[0], depth + 1, node, 0);
        break;
      case Node::If:
        os << "|- scf.if";
        printRegion(node, anchor);
        printPrefix(depth + 1);
        os << "|- then\n";
        printChain(node->children[0], depth + 2, node, 0);
        printPrefix(depth + 1);
        os << "|- else"
           << (cast<scf::IfOp>(node->op).elseBlock() ? "" : " (virtual)")
           << "\n";
        printChain(node->children[1], depth + 2, node, 1);
        break;
      case Node::Enter:
      case Node::Exit:
        os << (node->kind == Node::Enter ? "|- ENTER" : "|- EXIT");
        printPieces(node, anchor);
        if (node->kind == Node::Exit)
          printYield(parent, index);
        os << "\n";
        break;
      case Node::Func:
        llvm_unreachable("function node cannot occur in a child chain");
      }
    }
  }

  GroupDag &g;
  llvm::raw_ostream &os = llvm::errs();
};
} // namespace

bool shouldDumpDag() {
  const char *env = ::getenv("NVWS_INSERT_SEMA_DUMP_DAG");
  return env && StringRef(env) == "1";
}

void dumpSyncDagTree(GroupDag &g) {
  if (shouldDumpDag())
    SyncDagDumper(g).printTree();
}

void dumpSyncDagTrees(MutableArrayRef<GroupDag> groups) {
  for (GroupDag &g : groups)
    dumpSyncDagTree(g);
}

void dumpSyncDags(MutableArrayRef<GroupDag> groups, triton::FuncOp func) {
  if (!shouldDumpDag())
    return;
  llvm::errs() << "==== NVWS InsertSemas SYNC-DAG ====\n";
  llvm::errs() << "function: @" << func.getName() << "\n";
  llvm::errs() << "groups: " << groups.size() << "\n";
  for (GroupDag &g : groups)
    SyncDagDumper(g).print(func);
}

struct EmitCtx {
  triton::FuncOp func;
  Value poison; // the single function-level ub.poison token (contract E)
  Type tokenType;
  struct Slot {
    const GroupDag *g;
    unsigned index; // absolute result / iter_arg index in the NEW op
  };
  llvm::MapVector<Operation *, SmallVector<Slot, 2>> slots;
  DenseMap<const GroupDag *, SmallVector<Value>> backings;
  DenseMap<const Sema *, Value> semaphores;
  DenseMap<const Node *, Operation *> nodeOps;
  DenseSet<Operation *> exactReuseBufferOps;
  struct CachedReuseContract {
    Value view;
    Value token;
  };
  SmallVector<CachedReuseContract, 2> cachedReuseContracts;

  void index(const GroupDag &g) {
    nodeOps[g.root] = g.root->op;
    for (Node *head : g.root->children)
      forEachNode(head, [&](Node *node) { nodeOps[node] = node->op; });
  }
  Operation *op(const Node *node) const {
    assert(nodeOps.count(node) && "unindexed SYNC-DAG node");
    return nodeOps.lookup(node);
  }
  void replaceOp(Operation *oldOp, Operation *newOp) {
    for (auto &entry : nodeOps)
      if (entry.second == oldOp)
        entry.second = newOp;
  }
  ArrayRef<Value> backing(const GroupDag &g) const {
    return backings.find(g.physicalBacking)->second;
  }
  Value semaphore(const GroupDag &g, SemaId id) const {
    return semaphores.lookup(&g.semas[id]);
  }
};

static bool sameViewType(Type lhs, Type rhs);

struct RenderState {
  struct Token {
    Value value;
    Value sema;
    TokenRef ref;
  };
  SmallVector<Token, 2> tokens;
  struct ViewBundle {
    Node *producer = nullptr;
    SemaId channel = 0;
    Value token;
    Value semaphore;
    Owner owner;
    std::optional<int64_t> bufferStageOffset;
    SmallVector<Value, 2> buffers;
  };
  std::optional<ViewBundle> view;
  void clearViews() { view.reset(); }
  ViewBundle *findViewBundle(const Token &source, const Owner &owner,
                             std::optional<int64_t> bufferStageOffset,
                             MemberId member, Type resultType) {
    if (!view || view->producer != source.ref.producer ||
        view->channel != source.ref.sema || view->token != source.value ||
        view->semaphore != source.sema || !sameOwner(view->owner, owner) ||
        view->bufferStageOffset != bufferStageOffset ||
        !sameViewType(view->buffers[member].getType(), resultType))
      return nullptr;
    return &*view;
  }
  const Token *tokenForSource(const Node *producer) const {
    for (const Token &token : tokens)
      if (token.ref.producer == producer)
        return &token;
    return nullptr;
  }
  void recordToken(Value value, Value sema, const TokenRef &ref) {
    llvm::erase_if(tokens, [&](const Token &token) {
      return token.ref.producer == ref.producer;
    });
    tokens.push_back(Token{value, sema, ref});
  }
  RenderState nested() const {
    RenderState copy = *this;
    copy.clearViews();
    return copy;
  }
};
template <typename OpT, typename... Args>
static OpT emitInto(OpBuilder &b, Location loc, const Owner &owner,
                    gpu::StageCluster stageCluster, Args &&...args) {
  std::optional<SetVector<int>> ids = SetVector<int>();
  if (owner)
    ids->insert(owner->first);
  else
    ids = std::nullopt;
  auto op = gpu::createInto<OpT>(b, loc, ids, stageCluster,
                                 std::forward<Args>(args)...);
  if (owner) {
    auto forOp = op->template getParentOfType<scf::ForOp>();
    while (forOp && !gpu::hasWarpSpecializeTag(forOp))
      forOp = forOp->template getParentOfType<scf::ForOp>();
    if (!forOp) {
      if (owner->first == 0)
        op->removeAttr(gpu::kPartitionAttrName);
      else
        gpu::setWarpSpecializeTag(op, owner->second);
    }
  }
  return op;
}
static Operation *nextRealOp(const EmitCtx &ctx, const Node *n) {
  for (const Node *m = n; m; m = m->next)
    if ((m->kind == Node::Access || m->kind == Node::For ||
         m->kind == Node::If) &&
        ctx.op(m))
      return ctx.op(m);
  return nullptr;
}
static Block *chainBlock(const EmitCtx &ctx, Node *node) {
  if (!node || !node->parent || !ctx.op(node->parent))
    return nullptr;
  Node *head = node;
  while (head->prev)
    head = head->prev;
  Node *parent = node->parent;
  if (parent->kind == Node::Func)
    return &ctx.op(parent)->getRegion(0).front();
  for (auto [index, child] : llvm::enumerate(parent->children))
    if (child == head && index < ctx.op(parent)->getNumRegions())
      return &ctx.op(parent)->getRegion(index).front();
  return nullptr;
}
static SetVector<int> partitionIdsOfFwd(Operation *op) {
  SetVector<int> s;
  if (gpu::hasPartition(op))
    for (int p : gpu::getPartitionIds(op))
      s.insert(p);
  return s;
}
static ArrayAttr asyncOpsAttr(MLIRContext *ctx, const Node *rel) {
  SmallVector<Attribute, 2> elems;
  for (AsyncOp p : rel->payloads)
    elems.push_back(nvws::AsyncOpAttr::get(ctx, p));
  return ArrayAttr::get(ctx, elems);
}
static Value materializeI32Before(Operation *op, int64_t value);
static void nukeGroupTokens(EmitCtx &ctx, const GroupDag &g) {
  auto nukeOp = [&](Operation *op) {
    Value token;
    if (auto load = dyn_cast<nvidia_gpu::TMEMLoadOp>(op)) {
      load.getDepMutable().clear();
      token = load.getToken();
    } else if (auto store = dyn_cast<nvidia_gpu::TMEMStoreOp>(op)) {
      store.getDepMutable().clear();
      token = store.getToken();
    } else if (auto mma = dyn_cast<nvidia_gpu::MMAv5OpInterface>(op)) {
      if (g.aliases.count(mma.getAccumulator())) {
        mma.getAccDepMutable().clear();
        token = mma.getToken();
      }
    } else if (auto alloc = dyn_cast<nvidia_gpu::TMEMAllocOp>(op)) {
      token = alloc.getToken();
    }
    if (token)
      token.replaceAllUsesWith(ctx.poison);
  };
  for (const Member &m : g.pieceTable.members)
    nukeOp(m.allocOp);
  for (Operation *op : g.accessNodeOps)
    nukeOp(op);
}
static bool isScalesEnc(gpu::MemDescType t) {
  return isa<nvidia_gpu::TensorMemoryScalesEncodingAttr>(t.getEncoding());
}
static gpu::MemDescType withMutable(gpu::MemDescType t, bool m) {
  if (t.getMutableMemory() == m)
    return t;
  return gpu::MemDescType::get(t.getShape(), t.getElementType(),
                               t.getEncoding(), t.getMemorySpace(), m,
                               t.getAllocShape());
}
static gpu::MemDescType genericViewType(gpu::MemDescType backing) {
  auto shape = backing.getShape();
  return gpu::MemDescType::get(
      isScalesEnc(backing) ? shape : shape.drop_front(),
      backing.getElementType(), backing.getEncoding(), backing.getMemorySpace(),
      /*mutableMemory=*/true, backing.getShape());
}
static bool sameViewType(Type a, Type b) {
  auto x = cast<gpu::MemDescType>(a), y = cast<gpu::MemDescType>(b);
  return x.getShape() == y.getShape() &&
         x.getElementType() == y.getElementType() &&
         x.getEncoding() == y.getEncoding() &&
         x.getMemorySpace() == y.getMemorySpace() &&
         x.getMutableMemory() == y.getMutableMemory();
}
static gpu::MemDescType viewType(const GroupDag &g, MemberId member,
                                 const Touch &touch, gpu::MemDescType backing) {
  if (g.isTmem() || touch.member != member)
    return genericViewType(backing);
  Type type = g.pieceTable.members[member].type;
  if (touch.alias.empty())
    type = touch.accessType;
  for (const AliasStep &step : touch.alias) {
    if (step.op->getName().getStringRef() != "ttg.memdesc_index")
      break;
    type = step.resultType;
  }
  return withMutable(cast<gpu::MemDescType>(type), true);
}

static Value emitBacking(OpBuilder &b, Location loc, const GroupDag &g,
                         const Member &member) {
  auto type = backingType(g, member);
  Value backing;
  if (g.isTmem())
    backing =
        nvidia_gpu::TMEMAllocOp::create(b, loc, type, Value()).getResult();
  else
    backing = gpu::LocalAllocOp::create(b, loc, type).getResult();
  for (StringRef name :
       {kBufferIdAttrName, kBufferOffsetAttrName, kBufferCopyAttrName,
        kBufferCircularAttrName, kBufferStartAttrName})
    if (Attribute attr = member.allocOp->getAttr(name))
      backing.getDefiningOp()->setAttr(name, attr);
  return backing;
}
static Value emitTmemView(OpBuilder &b, Location loc, Value owner,
                          gpu::MemDescType target, int64_t offset,
                          int64_t sizeHint, bool reinterpret = false) {
  Value view = nvidia_gpu::TMEMSubSliceOp::create(
      b, loc, owner, static_cast<int32_t>(offset),
      static_cast<int32_t>(sizeHint));
  if (reinterpret || view.getType() != target)
    view = gpu::MemDescReinterpretOp::create(b, loc, target, view);
  return view;
}
static Value emitMixedCopyLocalView(OpBuilder &b, Location loc, Value owner,
                                    const GroupDag &g,
                                    const Member &member) {
  SmallVector<int64_t> logicalShape(member.type.getShape());
  logicalShape.insert(logicalShape.begin(), member.copies);
  auto logicalType = gpu::MemDescType::get(
      logicalShape, member.type.getElementType(), member.type.getEncoding(),
      member.type.getMemorySpace(), /*mutableMemory=*/true);
  Value logical =
      gpu::MemDescReinterpretOp::create(b, loc, logicalType, owner);
  Value slot = arith::ConstantIntOp::create(
      b, loc, member.circularStart, 32);
  Value view = gpu::MemDescIndexOp::create(b, loc, member.type, logical, slot);
  return gpu::MemDescReinterpretOp::create(b, loc, backingType(g, member),
                                            view);
}
static void materializeLogicalBacking(EmitCtx &ctx, const GroupDag &g) {
  auto &members = g.pieceTable.members;
  Operation *anchor = g.physicalBacking->physicalBackingAnchor;
  OpBuilder b(anchor);
  Location loc = anchor->getLoc();
  SmallVector<Value> &backing = ctx.backings[&g];
  backing.resize(members.size());
  for (auto [i, member] : llvm::enumerate(members)) {
    if (member.backingPrimary != i)
      continue;
    backing[i] = emitBacking(b, loc, g, member);
    for (unsigned j = 0; j < members.size(); ++j)
      if (j != i && members[j].backingPrimary == i && !g.isTmem() &&
          members[j].allocOp->hasAttr(kBufferStartAttrName))
        backing[j] = members[j].copies == g.numCopies
                         ? gpu::MemDescReinterpretOp::create(
                               b, loc, backingType(g, members[j]), backing[i])
                         : emitMixedCopyLocalView(b, loc, backing[i], g,
                                                  members[j]);
    for (int j = members.size() - 1; j >= 0; --j)
      if (j != i && members[j].backingPrimary == i) {
        if (!g.isTmem() &&
            members[j].allocOp->hasAttr(kBufferStartAttrName)) {
          continue;
        }
        auto target = backingType(g, members[j]);
        if (members[j].offset == member.offset &&
            target == backing[i].getType())
          backing[j] = backing[i];
        else
          backing[j] = emitTmemView(b, loc, backing[i], target,
                                    members[j].offset - member.offset,
                                    target.getShape().back());
      }
  }
}
static void emitPhysicalIR(EmitCtx &ctx, ArrayRef<const GroupDag *> groups) {
  for (const GroupDag *group : groups) {
    const GroupDag &g = *group;
    if (!ctx.backings.count(g.physicalBacking)) {
      if (g.isCircular()) {
        const GroupDag &owner = *g.physicalBacking;
        OpBuilder b(g.physicalBacking->physicalBackingAnchor);
        ctx.backings[&owner].assign(
            1, emitBacking(b, owner.semaAnchor->getLoc(), owner,
                           owner.pieceTable.members.front()));
      } else {
        materializeLogicalBacking(ctx, g);
      }
    }
    OpBuilder b(ctx.func);
    Operation *anchor = g.semaAnchor;
    b.setInsertionPoint(anchor);
    SmallVector<Type> baseTypes;
    for (const Member &member : g.pieceTable.members)
      baseTypes.push_back(backingType(g, member));
    auto semaTy = nvws::SemaphoreType::get(
        b.getContext(), nvws::TypeArrayAttr::get(b.getContext(), baseTypes));
    for (bool entry : {true, false})
      for (const Sema &s : g.semas) {
        if (s.entryOwner.has_value() != entry)
          continue;
        if (s.physicalPrimary != &s) {
          ctx.semaphores[&s] = ctx.semaphores.lookup(s.physicalPrimary);
          continue;
        }
        uint32_t releasedMask = s.releasedMask.value_or(
            entry ? nvws::getAllReleasedMask() : 0);
        auto create = nvws::SemaphoreCreateOp::create(
            b, anchor->getLoc(), semaTy, ctx.backing(g), releasedMask);
        create.setPendingCountAttr(b.getI32IntegerAttr(s.count));
        ctx.semaphores[&s] = create.getResult();
      }
  }
}

static void materializeTokenlessMembers(EmitCtx &ctx,
                                        ArrayRef<const GroupDag *> groups) {
  for (const GroupDag *group : groups) {
    if (!group->semas.empty())
      continue;
    ArrayRef<Value> backing = ctx.backing(*group);
    for (auto [index, member] : llvm::enumerate(group->pieceTable.members)) {
      OpBuilder b(member.allocOp);
      Value zero = arith::ConstantIntOp::create(b, member.allocOp->getLoc(), 0,
                                                32);
      auto view = gpu::MemDescIndexOp::create(
          b, member.allocOp->getLoc(), member.type, backing[index], zero);
      for (StringRef name : {"async_task_id", gpu::kPartitionAttrName,
                             gpu::kWarpSpecializeTagAttrName, "loop.stage",
                             "loop.cluster"})
        if (Attribute attr = member.allocOp->getAttr(name)) {
          zero.getDefiningOp()->setAttr(name, attr);
          view->setAttr(name, attr);
        }
      member.allocOp->getResult(0).replaceAllUsesWith(view.getResult());
      member.allocOp->erase();
    }
  }
}

static void rewriteRegionLayouts(EmitCtx &ctx, ArrayRef<GroupDag> groups) {
  struct Want {
    const GroupDag *g;
    Owner owner;
  };
  llvm::MapVector<Operation *, SmallVector<Want, 2>> wanted;
  for (const GroupDag &g : groups)
    for (Node *head : g.root->children)
      forEachNode(head, [&](Node *n) {
        if (n->flow)
          wanted[ctx.op(n)].push_back(Want{&g, n->flow->owner});
      });

  std::function<bool(Value, DenseSet<Value> &)> hasRealUse =
      [&](Value value, DenseSet<Value> &seen) -> bool {
    if (!seen.insert(value).second)
      return false;
    for (OpOperand &use : value.getUses()) {
      Operation *owner = use.getOwner();
      if (auto yield = dyn_cast<scf::YieldOp>(owner)) {
        Operation *parent = yield->getParentOp();
        unsigned index = use.getOperandNumber();
        if (index < parent->getNumResults() &&
            hasRealUse(parent->getResult(index), seen))
          return true;
        continue;
      }
      if (auto nested = dyn_cast<scf::ForOp>(owner)) {
        unsigned operand = use.getOperandNumber();
        if (operand >= 3) {
          unsigned index = operand - 3;
          if (index < nested.getNumRegionIterArgs() &&
              (hasRealUse(nested.getRegionIterArg(index), seen) ||
               hasRealUse(nested.getResult(index), seen)))
            return true;
          continue;
        }
      }
      return true;
    }
    return false;
  };

  struct RegionLayout {
    llvm::BitVector drop;
    ArrayRef<Want> add;
  };
  llvm::MapVector<Operation *, RegionLayout> layouts;
  ctx.func.walk([&](Operation *op) {
    if (!isa<scf::ForOp, scf::IfOp>(op))
      return;
    llvm::BitVector drop(op->getNumResults());
    for (auto [i, result] : llvm::enumerate(op->getResults())) {
      if (result.getType() != ctx.tokenType)
        continue;
      DenseSet<Value> seen;
      bool live = hasRealUse(result, seen);
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        seen.clear();
        live |= hasRealUse(forOp.getRegionIterArg(i), seen);
      }
      if (!live)
        drop.set(i);
    }
    auto it = wanted.find(op);
    ArrayRef<Want> add =
        it == wanted.end() ? ArrayRef<Want>() : ArrayRef<Want>(it->second);
    if (drop.any() || !add.empty())
      layouts.insert({op, RegionLayout{std::move(drop), add}});
  });

  SmallVector<Operation *> ops;
  for (auto &[op, _] : layouts)
    ops.push_back(op);
  llvm::stable_sort(ops, [](Operation *a, Operation *b) {
    auto depth = [](Operation *op) {
      unsigned d = 0;
      while ((op = op->getParentOp()))
        ++d;
      return d;
    };
    return depth(a) < depth(b);
  });
  OpBuilder b(ctx.func);
  for (Operation *op : ops) {
    RegionLayout &layout = layouts[op];
    ArrayRef<Want> list = layout.add;
    unsigned nSlots = list.size();
    SmallVector<Value> poisons(nSlots, ctx.poison);
    Operation *newOp = nullptr;
    unsigned base = op->getNumResults() - layout.drop.count();
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      SmallVector<Value> inits;
      for (auto [i, init] : llvm::enumerate(forOp.getInits()))
        if (!layout.drop.test(i))
          inits.push_back(init);
      inits.append(poisons);
      b.setInsertionPoint(forOp);
      auto newFor =
          scf::ForOp::create(b, forOp.getLoc(), forOp.getLowerBound(),
                             forOp.getUpperBound(), forOp.getStep(), inits);
      newFor->setAttrs(forOp->getAttrs());
      newFor.getRegion().takeBody(forOp.getRegion());
      Block &body = newFor.getRegion().front();
      llvm::BitVector dropArgs(body.getNumArguments());
      for (unsigned i : layout.drop.set_bits()) {
        body.getArgument(1 + i).replaceAllUsesWith(ctx.poison);
        dropArgs.set(1 + i); // +1: induction variable
      }
      body.eraseArguments(dropArgs);
      for (unsigned i = 0; i < nSlots; ++i)
        body.addArgument(ctx.tokenType, ctx.poison.getLoc());
      newOp = newFor;
    } else {
      auto ifOp = cast<scf::IfOp>(op);
      SmallVector<Type> types;
      for (auto [i, result] : llvm::enumerate(ifOp.getResults()))
        if (!layout.drop.test(i))
          types.push_back(result.getType());
      types.append(nSlots, ctx.tokenType);
      b.setInsertionPoint(ifOp);
      bool withElse = !ifOp.getElseRegion().empty() || nSlots != 0;
      auto newIf =
          scf::IfOp::create(b, ifOp.getLoc(), types, ifOp.getCondition(),
                            /*withElseRegion=*/withElse);
      newIf->setAttrs(ifOp->getAttrs());
      newIf.getThenRegion().takeBody(ifOp.getThenRegion());
      if (!ifOp.getElseRegion().empty())
        newIf.getElseRegion().takeBody(ifOp.getElseRegion());
      scf::IfOp::ensureTerminator(newIf.getThenRegion(), b, ifOp.getLoc());
      if (withElse)
        scf::IfOp::ensureTerminator(newIf.getElseRegion(), b, ifOp.getLoc());
      newOp = newIf;
    }

    for (Region &region : newOp->getRegions()) {
      if (region.empty())
        continue;
      auto yield = cast<scf::YieldOp>(region.front().getTerminator());
      SmallVector<Value> operands;
      for (auto [i, operand] : llvm::enumerate(yield.getOperands()))
        if (!layout.drop.test(i))
          operands.push_back(operand);
      operands.append(poisons);
      OpBuilder yb(yield);
      auto newYield = scf::YieldOp::create(yb, yield.getLoc(), operands);
      // appendToForOpYield historically dropped loop-yield metadata before
      // the parent partition was stamped below. Preserve that exact behavior
      // for planned loop slots; drop-only loops and all ifs retain attributes.
      if (nSlots == 0 || isa<scf::IfOp>(newOp))
        newYield->setAttrs(yield->getAttrs());
      yield.erase();
    }

    if (auto attr =
            newOp->getAttrOfType<ArrayAttr>(gpu::kPartitionOutputsAttrName);
        attr && attr.size() == layout.drop.size()) {
      SmallVector<Attribute> kept;
      for (auto [i, value] : llvm::enumerate(attr.getValue()))
        if (!layout.drop.test(i))
          kept.push_back(value);
      newOp->setAttr(gpu::kPartitionOutputsAttrName,
                     ArrayAttr::get(newOp->getContext(), kept));
    }
    SmallVector<SetVector<int>, 4> outputs;
    if (nSlots != 0 && gpu::hasPartition(newOp)) {
      if (base != 0 && newOp->hasAttr(gpu::kPartitionOutputsAttrName))
        outputs = gpu::getPartitionOutputs(newOp);
      else if (base != 0)
        outputs.assign(base, partitionIdsOfFwd(newOp));
    }
    for (const Want &w : list) {
      SetVector<int> set;
      if (w.owner)
        set.insert(w.owner->first);
      else
        set = partitionIdsOfFwd(newOp);
      outputs.push_back(set);
    }
    if (nSlots != 0 && gpu::hasPartition(newOp)) {
      gpu::setPartitionOutputs(newOp, outputs);
      auto ids = gpu::getPartitionIds(newOp);
      for (Region &r : newOp->getRegions())
        for (Block &blk : r)
          if (Operation *term = blk.getTerminator();
              term && !gpu::hasPartition(term))
            term->setAttr(gpu::kPartitionAttrName,
                          DenseI32ArrayAttr::get(
                              newOp->getContext(),
                              SmallVector<int>(ids.begin(), ids.end())));
    }
    unsigned next = 0;
    for (auto [i, result] : llvm::enumerate(op->getResults())) {
      if (layout.drop.test(i))
        result.replaceAllUsesWith(ctx.poison);
      else
        result.replaceAllUsesWith(newOp->getResult(next++));
    }
    ctx.replaceOp(op, newOp);
    op->erase();
    auto &rec = ctx.slots[newOp];
    for (auto [i, w] : llvm::enumerate(list))
      rec.push_back(EmitCtx::Slot{w.g, base + static_cast<unsigned>(i)});
  }
}
static unsigned slotIndexFor(EmitCtx &ctx, Operation *op, const GroupDag *g) {
  for (const EmitCtx::Slot &s : ctx.slots[op])
    if (s.g == g)
      return s.index;
  llvm_unreachable("missing slot");
}
static void refreshAliasResultTypes(Operation *op, Value source) {
  // A mapped semaphore view can carry a staged allocShape that was absent from
  // the original alias operand. Let each alias op derive its result from the
  // cloned operands before falling back to the old mutability-only adjustment.
  if (auto typeInfer = dyn_cast<InferTypeOpInterface>(op)) {
    SmallVector<Type> inferredTypes;
    if (succeeded(typeInfer.inferReturnTypes(
            op->getContext(), op->getLoc(), op->getOperands(),
            op->getAttrDictionary(), op->getPropertiesStorage(),
            op->getRegions(), inferredTypes)) &&
        inferredTypes.size() == op->getNumResults()) {
      for (auto [result, type] : llvm::zip(op->getResults(), inferredTypes))
        result.setType(type);
      return;
    }
  }
  op->getResult(0).setType(
      withMutable(cast<gpu::MemDescType>(op->getResult(0).getType()),
                  cast<gpu::MemDescType>(source.getType()).getMutableMemory()));
}
static Value getView(EmitCtx &ctx, const GroupDag &g, RenderState &rs,
                     Node *node, const Touch &touch, Operation *accessOp,
                     const Owner &owner, const RenderState::Token &source) {
  SmallVector<Type, 2> types;
  ArrayRef<Value> backing = ctx.backing(g);
  for (auto [mi, m] : llvm::enumerate(g.pieceTable.members)) {
    auto bt = cast<gpu::MemDescType>(backing[mi].getType());
    types.push_back(viewType(g, static_cast<MemberId>(mi), touch, bt));
  }
  gpu::StageCluster stageCluster = gpu::getStageCluster(accessOp);
  RenderState::ViewBundle *bundle =
      rs.findViewBundle(source, owner, node->bufferStageOffset, touch.member,
                        types[touch.member]);
  if (bundle && node->releasedViewRelease) {
    Value buffer = bundle->buffers[touch.member];
    if (!ctx.exactReuseBufferOps.contains(buffer.getDefiningOp()))
      ctx.cachedReuseContracts.push_back({buffer, source.value});
  }
  if (!bundle) {
    OpBuilder b(accessOp);
    auto buf = emitInto<nvws::SemaphoreBufferOp>(
        b, accessOp->getLoc(), owner, stageCluster, source.sema,
        TypeRange(types), source.value);
    if (node->releasedViewRelease)
      ctx.exactReuseBufferOps.insert(buf.getOperation());
    if (node->bufferStageOffset)
      buf.setStage(materializeI32Before(buf, *node->bufferStageOffset));
    rs.view =
        RenderState::ViewBundle{source.ref.producer,
                                source.ref.sema,
                                source.value,
                                source.sema,
                                owner,
                                node->bufferStageOffset,
                                SmallVector<Value, 2>(buf.getBuffers().begin(),
                                                      buf.getBuffers().end())};
    bundle = &*rs.view;
  }
  Value base = bundle->buffers[touch.member];
  Value cur = base;
  OpBuilder b(accessOp);
  for (const AliasStep &step : touch.alias) {
    Operation *old = step.op;
    if (old->getName().getStringRef() == "ttg.memdesc_index" &&
        sameViewType(step.resultType, cur.getType()))
      continue;
    IRMapping mapping;
    for (auto [idx, operand] : llvm::enumerate(old->getOperands()))
      mapping.map(operand, idx == step.operandIdx ? cur : operand);
    Value source = cur;
    Operation *cloned = b.clone(*old, mapping);
    refreshAliasResultTypes(cloned, source);
    cur = cloned->getResult(0);
  }
  return cur;
}

static LogicalResult renderChain(EmitCtx &ctx, const GroupDag &g, Node *head,
                                 RenderState &rs);
static Operation *renderAccess(EmitCtx &ctx, const GroupDag &g, Node *n,
                               RenderState &rs,
                               const RenderState::Token &source) {
  Operation *op = ctx.op(n);
  Operation *anchor = n->completionAnchor ? n->completionAnchor : op;
  for (const Touch &touch : n->touches) {
    Value view = getView(ctx, g, rs, n, touch, op, n->owner, source);
    if (auto ta = dyn_cast<nvidia_gpu::TMEMAllocOp>(op)) {
      OpBuilder b(op);
      auto pidsc = std::make_pair(n->owner, gpu::getStageCluster(op));
      auto vTrue = emitInto<arith::ConstantOp>(
          b, op->getLoc(), n->owner, pidsc.second, b.getBoolAttr(true));
      anchor = emitInto<nvidia_gpu::TMEMStoreOp>(b, op->getLoc(), n->owner,
                                                 pidsc.second, Type(), view,
                                                 Value(), ta.getSrc(), vTrue);
      ta.getResult().replaceUsesWithIf(view, [&](OpOperand &use) {
        return !isa<nvws::SemaphoreCreateOp>(use.getOwner()) &&
               use.getOwner() != view.getDefiningOp() &&
               !g.accessNodeOps.contains(use.getOwner());
      });
      return anchor;
    }
    if (auto la = dyn_cast<gpu::LocalAllocOp>(op)) {
      OpBuilder b(op);
      Value src = la.getSrc();
      if (src && !isa<RankedTensorType>(src.getType())) {
        auto splat = emitInto<triton::SplatOp>(
            b, op->getLoc(), n->owner, gpu::getStageCluster(op),
            RankedTensorType::get(
                cast<gpu::MemDescType>(view.getType()).getShape(),
                src.getType()),
            src);
        src = splat.getResult();
      }
      anchor = emitInto<gpu::LocalStoreOp>(b, op->getLoc(), n->owner,
                                           gpu::getStageCluster(op), src, view);
      la.getResult().replaceUsesWithIf(view, [&](OpOperand &use) {
        return !isa<nvws::SemaphoreCreateOp>(use.getOwner()) &&
               !g.accessNodeOps.contains(use.getOwner());
      });
      return anchor;
    }
    for (OpOperand &o : op->getOpOperands())
      if (o.get() == touch.accessValue)
        o.set(view);
  }
  return anchor;
}

static void recordProducerAlias(RenderState &state, Node *producer,
                                Node *except, const RenderState::Token &token) {
  if (producer && producer != except)
    state.recordToken(token.value, token.sema,
                      TokenRef{producer, token.ref.sema, token.ref.owner});
}
static const RenderState::Token *
regionExitToken(const RegionFlow &flow, unsigned branch, RenderState &state,
                const RenderState::Token *passThrough) {
  Node *final = flow.exits[branch];
  return final ? state.tokenForSource(final) : passThrough;
}
static LogicalResult
renderPlainLoop(EmitCtx &ctx, const GroupDag &g, Node *node, RenderState &state,
                const std::optional<RenderState::Token> &incoming) {
  RenderState body = state.nested();
  if (incoming)
    body.recordToken(incoming->value, incoming->sema,
                     TokenRef{node, incoming->ref.sema, incoming->ref.owner});
  if (failed(renderChain(ctx, g, node->children[0], body)))
    return failure();
  if (incoming)
    state.recordToken(incoming->value, incoming->sema,
                      TokenRef{node, incoming->ref.sema, incoming->ref.owner});
  state.clearViews();
  return success();
}
static LogicalResult renderCarriedLoop(EmitCtx &ctx, const GroupDag &g,
                                       Node *node, scf::ForOp forOp,
                                       RenderState &state,
                                       const RenderState::Token &incoming,
                                       const TokenRef &resultRef) {
  unsigned index = slotIndexFor(ctx, ctx.op(node), &g);
  forOp.getInitsMutable()[index].assign(incoming.value);
  RenderState body = state.nested();
  RenderState::Token carrier{forOp.getRegionIterArg(index),
                             ctx.semaphore(g, resultRef.sema), resultRef};
  body.recordToken(carrier.value, carrier.sema, carrier.ref);
  recordProducerAlias(body, node->tokenSource, node, carrier);
  if (failed(renderChain(ctx, g, node->children[0], body)))
    return failure();
  const RegionFlow &flow = *node->flow;
  const RenderState::Token *bodyToken =
      regionExitToken(flow, 0, body, body.tokenForSource(resultRef.producer));
  if (!bodyToken)
    return semaError(ctx.op(node))
           << "loop body exports no exact carried token";
  cast<scf::YieldOp>(forOp.getBody()->getTerminator())
      .setOperand(index, bodyToken->value);
  RenderState::Token result{forOp.getResult(index),
                            ctx.semaphore(g, resultRef.sema), resultRef};
  state.recordToken(result.value, result.sema, result.ref);
  recordProducerAlias(state, node->tokenSource, node, result);
  recordProducerAlias(state, flow.exits.front(), node, result);
  state.clearViews();
  return success();
}

static LogicalResult renderRegion(EmitCtx &ctx, const GroupDag &g, Node *n,
                                  RenderState &rs) {
  Operation *op = ctx.op(n);
  std::optional<RenderState::Token> incoming;
  if (n->tokenSource) {
    const RenderState::Token *source = rs.tokenForSource(n->tokenSource);
    if (!source)
      return semaError(op)
             << "region cannot resolve its incoming token producer";
    incoming = *source;
  }
  TokenRef resultRef;
  if (n->flow) {
    bool needsIncoming =
        n->kind == Node::For || llvm::is_contained(n->flow->exits, nullptr);
    if (incoming && incoming->ref.owner &&
        !sameOwner(incoming->ref.owner, n->flow->owner))
      return semaError(op)
             << "threaded region input belongs to another partition";
    if (needsIncoming && !incoming)
      return semaError(op)
             << "pass-through region has no exact incoming token producer";
    assert(!n->flow->inheritsInputChannel ||
           (incoming && incoming->ref.sema == *n->flow->sema));
    resultRef = TokenRef{n, *n->flow->sema, n->flow->owner};
  }
  if (!n->requiredParts.empty() && gpu::hasPartition(op)) {
    SetVector<int> set = gpu::getPartitionIds(op);
    unsigned before = set.size();
    for (int p : n->requiredParts)
      set.insert(p);
    if (set.size() != before)
      gpu::setPartition(op, set.getArrayRef());
  }
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    if (!n->flow)
      return renderPlainLoop(ctx, g, n, rs, incoming);
    return renderCarriedLoop(ctx, g, n, forOp, rs, *incoming, resultRef);
  }
  auto ifOp = cast<scf::IfOp>(op);
  if (!ifOp.elseBlock()) {
    Block &block = ifOp.getElseRegion().emplaceBlock();
    OpBuilder b = OpBuilder::atBlockEnd(&block);
    scf::YieldOp::create(b, ifOp.getLoc());
  }
  RenderState thenSt = rs.nested(), elseSt = rs.nested();
  if (incoming) {
    TokenRef boundary{n, incoming->ref.sema, incoming->ref.owner};
    thenSt.recordToken(incoming->value, incoming->sema, boundary);
    elseSt.recordToken(incoming->value, incoming->sema, boundary);
  }
  if (failed(renderChain(ctx, g, n->children[0], thenSt)))
    return failure();
  if (failed(renderChain(ctx, g, n->children[1], elseSt)))
    return failure();
  if (n->flow) {
    const RegionFlow &c = *n->flow;
    unsigned idx = slotIndexFor(ctx, op, &g);
    const RenderState::Token *passThrough = incoming ? &*incoming : nullptr;
    const RenderState::Token *thenToken =
        regionExitToken(c, 0, thenSt, passThrough);
    const RenderState::Token *elseToken =
        regionExitToken(c, 1, elseSt, passThrough);
    if (!thenToken || !elseToken)
      return semaError(op) << "if path exports no exact compatible token";
    auto thenYield = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    thenYield->setOperand(idx, thenToken->value);
    auto elseYield = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
    elseYield->setOperand(idx, elseToken->value);
    RenderState::Token result{ifOp.getResult(idx),
                              ctx.semaphore(g, resultRef.sema), resultRef};
    rs.recordToken(result.value, result.sema, result.ref);
    for (Node *final : c.exits)
      recordProducerAlias(rs, final, n, result);
  }
  rs.clearViews();
  return success();
}

static LogicalResult renderChain(EmitCtx &ctx, const GroupDag &g, Node *head,
                                 RenderState &rs) {
  Operation *lastReal = nullptr;
  for (Node *n = head; n; n = n->next) {
    const RenderState::Token *source = nullptr;
    if (n->kind == Node::Release || n->kind == Node::Access) {
      source = n->tokenSource ? rs.tokenForSource(n->tokenSource) : nullptr;
      if (!source)
        return semaError(ctx.op(n) ? ctx.op(n) : ctx.op(g.root))
               << "buffer consumer cannot resolve its exact token producer";
      if (source->ref.owner && !sameOwner(n->owner, source->ref.owner))
        return semaError(ctx.op(n) ? ctx.op(n) : ctx.op(g.root))
               << "buffer consumer token belongs to another partition";
    }
    switch (n->kind) {
    case Node::Enter:
    case Node::Exit:
      break; // markers; yield wiring is the parent's job
    case Node::Acquire: {
      assert(n->producedTokenOwner && "acquire token owner must be sealed");
      Operation *before = nextRealOp(ctx, n->next);
      OpBuilder b(ctx.func);
      if (before)
        b.setInsertionPoint(before);
      else if (lastReal &&
               !isa<triton::FuncOp>(lastReal->getBlock()->getParentOp())) {
        b.setInsertionPoint(lastReal->getBlock()->getTerminator());
      } else if (lastReal) {
        b.setInsertionPointAfter(lastReal);
      } else if (Block *block = chainBlock(ctx, n)) {
        b.setInsertionPoint(block->getTerminator());
      } else {
        return semaError(ctx.func) << "acquire has no exact containing block";
      }
      auto acq = emitInto<nvws::SemaphoreAcquireOp>(
          b, before ? before->getLoc() : ctx.func.getLoc(), n->owner,
          n->stageCluster, ctx.semaphore(g, n->sema), ctx.tokenType);
      if (n->stageOffset)
        acq.setStage(materializeI32Before(acq, *n->stageOffset));
      rs.recordToken(acq.getToken(), ctx.semaphore(g, n->sema),
                     TokenRef{n, n->sema, *n->producedTokenOwner});
      rs.clearViews();
      lastReal = acq;
      break;
    }
    case Node::Release: {
      OpBuilder b(ctx.func);
      if (lastReal)
        b.setInsertionPointAfter(lastReal);
      else if (Block *block = chainBlock(ctx, n))
        b.setInsertionPointToStart(block);
      else
        return semaError(ctx.func) << "release has no exact containing block";
      auto rel = emitInto<nvws::SemaphoreReleaseOp>(
          b, lastReal ? lastReal->getLoc() : ctx.func.getLoc(), n->owner,
          n->stageCluster, ctx.semaphore(g, n->sema), source->value,
          asyncOpsAttr(b.getContext(), n));
      if (n->stageOffset)
        rel.setStage(materializeI32Before(rel, *n->stageOffset));
      rel.setArriveCountAttr(b.getI32IntegerAttr(n->count));
      lastReal = rel;
      break;
    }
    case Node::Access:
      lastReal = renderAccess(ctx, g, n, rs, *source);
      break;
    case Node::For:
    case Node::If:
      if (failed(renderRegion(ctx, g, n, rs)))
        return failure();
      lastReal = ctx.op(n);
      break;
    case Node::Func:
      break;
    }
  }
  return success();
}

static Value materializeI32Before(Operation *op, int64_t value) {
  OpBuilder b(op);
  auto cst = emitInto<arith::ConstantOp>(b, op->getLoc(), resolveOwner(op),
                                         gpu::getStageCluster(op),
                                         b.getI32IntegerAttr(value));
  return cst.getResult();
}
static bool precedesInBlock(Operation *before, Operation *after) {
  return before->getBlock() == after->getBlock() &&
         before->isBeforeInBlock(after);
}
static LogicalResult verifyTokenLocality(ArrayRef<Operation *> operations) {
  auto idsOf = [](Operation *op) -> std::optional<SmallVector<int, 2>> {
    if (!gpu::hasPartition(op))
      return std::nullopt;
    auto set = gpu::getPartitionIds(op);
    SmallVector<int, 2> v(set.begin(), set.end());
    llvm::sort(v);
    return v;
  };
  LogicalResult result = success();
  auto trace = [&](Operation *consumer, Value root) {
    DenseSet<Value> seen;
    SmallVector<Value, 8> pending{root};
    auto consumerParts = idsOf(consumer);
    auto appendLoopInputs = [&](scf::ForOp loop, unsigned index) {
      pending.push_back(loop.getBody()->getTerminator()->getOperand(index));
      pending.push_back(loop.getInits()[index]);
    };
    while (!pending.empty()) {
      Value token = pending.pop_back_val();
      if (!seen.insert(token).second)
        continue;
      if (auto arg = dyn_cast<BlockArgument>(token)) {
        auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
        if (loop)
          appendLoopInputs(loop, arg.getArgNumber() - 1);
        continue; // Other block arguments are outside this verifier's scope.
      }
      Operation *def = token.getDefiningOp();
      if (!def || isa<ub::PoisonOp>(def))
        continue;
      if (auto acquire = dyn_cast<nvws::SemaphoreAcquireOp>(def)) {
        auto acquireParts = idsOf(acquire);
        if (!acquireParts)
          continue; // Root entry seed exemption.
        if (consumerParts && *acquireParts != *consumerParts) {
          InFlightDiagnostic diag =
              semaError(consumer)
              << "token-locality violation: token acquired in partition set "
                 "differs from consumer's";
          diag.attachNote(acquire.getLoc()) << "acquired here";
          result = failure();
          return;
        }
        continue;
      }
      unsigned index = cast<OpResult>(token).getResultNumber();
      if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
        pending.push_back(ifOp.elseYield().getOperand(index));
        pending.push_back(ifOp.thenYield().getOperand(index));
      } else if (auto forOp = dyn_cast<scf::ForOp>(def)) {
        appendLoopInputs(forOp, index);
      }
    }
  };
  for (Operation *op : operations)
    if (auto release = dyn_cast<nvws::SemaphoreReleaseOp>(op))
      trace(op, release.getToken());
  for (Operation *op : operations) {
    auto buffer = dyn_cast<nvws::SemaphoreBufferOp>(op);
    if (!buffer)
      continue;
    trace(op, buffer.getToken());
    auto bp = idsOf(buffer.getOperation());
    if (!bp)
      continue;
    for (Value view : buffer->getResults())
      for (Operation *user : view.getUsers()) {
        auto up = idsOf(user);
        if (up && *up != *bp) {
          InFlightDiagnostic diag =
              semaError(user)
              << "view-locality violation: view consumed outside its partition";
          diag.attachNote(buffer.getLoc()) << "view materialized here";
          result = failure();
        }
      }
  }
  return result;
}
static nvws::SemaphoreAcquireOp resolveAcquireThroughIfs(Value v) {
  while (auto ifOp = v.getDefiningOp<scf::IfOp>()) {
    unsigned idx = cast<OpResult>(v).getResultNumber();
    Value next = ifOp.thenYield()->getOperand(idx);
    for (Value yielded : {next, ifOp.elseYield()->getOperand(idx)})
      if (auto acquire = yielded.getDefiningOp<nvws::SemaphoreAcquireOp>())
        return acquire;
    v = next;
  }
  return v.getDefiningOp<nvws::SemaphoreAcquireOp>();
}
static LogicalResult
verifyEmittedIR(triton::FuncOp func,
                const DenseSet<Operation *> &exactReuseBufferOps,
                ArrayRef<EmitCtx::CachedReuseContract> cachedReuseContracts) {
  for (const EmitCtx::CachedReuseContract &contract : cachedReuseContracts) {
    auto buffer = contract.view.getDefiningOp<nvws::SemaphoreBufferOp>();
    if (!buffer || buffer.getToken() != contract.token)
      return semaError(func)
             << "emitter contract: malformed exact cached-view reuse";
    bool witnessed =
        llvm::any_of(contract.token.getUsers(), [&](Operation *tokenUser) {
          return isa<nvws::SemaphoreReleaseOp>(tokenUser) &&
                 precedesInBlock(buffer, tokenUser) &&
                 llvm::any_of(contract.view.getUsers(),
                              [&](Operation *viewUser) {
                                return precedesInBlock(tokenUser, viewUser);
                              });
        });
    if (!witnessed)
      return semaError(buffer) << "emitter contract: exact cached view has no "
                                  "use after its release";
  }
  SmallVector<Operation *> operations;
  func.walk([&](Operation *op) { operations.push_back(op); });
  for (Operation *op : operations) {
    if (!isa<scf::ForOp, scf::IfOp>(op) ||
        !op->hasAttr(gpu::kPartitionOutputsAttrName))
      continue;
    auto outputs = gpu::getPartitionOutputs(op);
    if (outputs.size() != op->getNumResults())
      return semaError(op) << "partition-outputs verifier: attribute has "
                           << outputs.size() << " entries for "
                           << op->getNumResults() << " results";
    SmallVector<Operation *, 2> terms;
    for (Region &region : op->getRegions())
      if (!region.empty())
        terms.push_back(region.front().getTerminator());
    for (auto [i, output] : llvm::enumerate(outputs))
      for (Operation *term : terms) {
        Operation *def = term->getOperand(i).getDefiningOp();
        if (!def || isa<ub::PoisonOp>(def) ||
            def->hasTrait<OpTrait::ConstantLike>() || !gpu::hasPartition(def))
          continue;
        SetVector<int> producer = gpu::getPartitionIds(def);
        if (producer.empty() ||
            llvm::any_of(producer, [&](int p) { return output.contains(p); }))
          continue;
        std::string have;
        llvm::raw_string_ostream os(have);
        for (int p : producer)
          os << p << " ";
        return semaError(op) << "partition-outputs verifier: result " << i
                             << " is produced by partition(s) " << os.str()
                             << "but ttg.partition.outputs names none of them";
      }
  }
  if (failed(verifyTokenLocality(operations)))
    return failure();

  auto checkToken = [&](Value token) -> LogicalResult {
    llvm::SmallDenseMap<Block *, SmallVector<Operation *, 4>> usersByBlock;
    for (Operation *user : token.getUsers())
      usersByBlock[user->getBlock()].push_back(user);
    for (auto &[block, users] : usersByBlock) {
      (void)block;
      llvm::sort(users, [](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
      });
      bool released = false;
      for (Operation *user : users) {
        released |= isa<nvws::SemaphoreReleaseOp>(user);
        if (released && isa<nvws::SemaphoreBufferOp>(user) &&
            !exactReuseBufferOps.contains(user))
          return semaError(user)
                 << "token has a buffer view after its release "
                    "(use-after-release; spec fable/semas-report3.md "
                    "Addendum B.3(b))";
      }
    }
    return success();
  };
  auto checkLoop = [&](scf::ForOp forOp) -> LogicalResult {
    llvm::SmallDenseMap<Value, unsigned> slotsPerBacking;
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    for (BlockArgument arg : forOp.getRegionIterArgs()) {
      if (!isa<gpu::AsyncTokenType>(arg.getType()))
        continue;
      if (failed(checkToken(arg)))
        return failure();
      unsigned idx = arg.getArgNumber() - 1; // skip induction variable
      nvws::SemaphoreAcquireOp acq =
          resolveAcquireThroughIfs(yieldOp.getOperand(idx));
      if (!acq)
        continue;
      auto create = acq.getSemaphore().getDefiningOp<nvws::SemaphoreCreateOp>();
      if (!create || create.getBuffers().empty())
        continue;
      Value backing = create.getBuffers().front();
      if (auto alloc = backing.getDefiningOp<gpu::LocalAllocOp>();
          alloc && alloc->hasAttr(kBufferCircularAttrName))
        continue;
      if (++slotsPerBacking[backing] > 1)
        return semaError(forOp)
               << "two token slots for one semaphore group in a single loop "
                  "(spec fable/semas-report3.md Addendum B.3(a)); "
                  "AssignStagePhase cannot thread this";
    }
    return success();
  };
  for (Operation *op : operations) {
    if (auto acquire = dyn_cast<nvws::SemaphoreAcquireOp>(op);
        acquire && failed(checkToken(acquire.getToken())))
      return failure();
    if (auto forOp = dyn_cast<scf::ForOp>(op);
        forOp && failed(checkLoop(forOp)))
      return failure();
  }
  return success();
}

LogicalResult emitIR(triton::FuncOp funcOp, ArrayRef<GroupDag> groups,
                     ArrayRef<ScheduleUpdate> updates) {
  for (const GroupDag &g : groups)
    assert(g.isSealed() && "EmitIR requires sealed SYNC-DAGs");
  OpBuilder scheduleBuilder(funcOp.getContext());
  for (auto [op, schedule] : updates)
    gpu::setStageCluster(scheduleBuilder, op, schedule);
  EmitCtx ctx;
  ctx.func = funcOp;
  ctx.tokenType = gpu::AsyncTokenType::get(funcOp.getContext());
  {
    OpBuilder b(&funcOp.getBody().front(), funcOp.getBody().front().begin());
    ctx.poison =
        ub::PoisonOp::create(b, funcOp.getLoc(), ctx.tokenType).getResult();
  }
  SmallVector<const GroupDag *> plannedGroups, activeGroups;
  for (const GroupDag &g : groups) {
    ctx.index(g);
    if (g.physicalBacking)
      plannedGroups.push_back(&g);
    if (!g.semas.empty())
      activeGroups.push_back(&g);
  }
  for (const GroupDag *group : activeGroups)
    nukeGroupTokens(ctx, *group);
  emitPhysicalIR(ctx, plannedGroups);
  materializeTokenlessMembers(ctx, plannedGroups);
  rewriteRegionLayouts(ctx, groups);
  for (const GroupDag *group : activeGroups) {
    RenderState rs;
    if (failed(renderChain(ctx, *group, group->root->children[0], rs)))
      return failure();
  }
  SmallVector<Operation *> aliasOps;
  ctx.func.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isSupportedAliasOp(op))
      aliasOps.push_back(op);
  });
  for (Operation *op : llvm::reverse(aliasOps))
    if (llvm::all_of(op->getResults(), [](Value v) { return v.use_empty(); }))
      op->erase();
  for (const GroupDag *group : activeGroups)
    for (const Member &m : group->pieceTable.members)
      if (m.allocOp && m.allocOp->getBlock() && m.allocOp->use_empty())
        m.allocOp->erase();
  // Managed buffer dependencies have all been rebuilt from exact semaphore
  // tokens.  A poison can remain only on detached legacy token plumbing whose
  // result is consumed by an unrelated opaque operation; keep the valid IR as
  // the pre-refactor emitter did.  Structural dead slots are removed above.
  if (ctx.poison.use_empty())
    ctx.poison.getDefiningOp()->erase();
  return verifyEmittedIR(funcOp, ctx.exactReuseBufferOps,
                         ctx.cachedReuseContracts);
}
} // namespace mlir::triton::nvws_semas
