// ACCESS analysis; see sema-docs/insert-semas/access-dag.md.
#include "InsertSemas.h"

namespace mlir::triton::nvws_semas {

FailureOr<SmallVector<GroupDag, 0>> collectGroups(triton::FuncOp funcOp) {
  llvm::MapVector<int64_t, SmallVector<Operation *, 2>> tmemBuckets, localBuckets;
  SmallVector<Operation *, 4> circularLocals;
  llvm::DenseSet<int64_t> syntheticIds; // negative keys = synthetic
  int64_t nextSynthetic = -1;
  LogicalResult result = success();
  funcOp.walk([&](Operation *op) {
    if (auto alloc = dyn_cast<nvidia_gpu::TMEMAllocOp>(op)) {
      std::optional<int64_t> id = getI64Attr(op, kBufferIdAttrName);
      int64_t key = id ? *id : nextSynthetic--;
      if (!id)
        syntheticIds.insert(key);
      tmemBuckets[key].push_back(op);
      return;
    }
    if (auto alloc = dyn_cast<gpu::LocalAllocOp>(op)) {
      auto type = cast<gpu::MemDescType>(alloc.getType());
      if (!type.getMutableMemory())
        return;
      std::optional<int64_t> id = getI64Attr(op, kBufferIdAttrName);
      if (op->hasAttr(kBufferCircularAttrName)) {
        if (!id) {
          result = semaError(op) << "circular local alloc requires buffer.id";
          return;
        }
        for (StringRef name : {kBufferCopyAttrName, kBufferStartAttrName})
          if (!op->hasAttr(name)) {
            result = semaError(op) << "circular local alloc requires " << name;
            return;
          }
        if (op->hasAttr(kBufferOffsetAttrName)) {
          result = semaError(op) << "circular local alloc must not carry buffer.offset";
          return;
        }
        circularLocals.push_back(op);
        return;
      }
      int64_t key = id ? *id : nextSynthetic--;
      if (!id)
        syntheticIds.insert(key);
      localBuckets[key].push_back(op);
      return;
    }
  });
  if (failed(result))
    return failure();
  SmallVector<GroupDag, 0> groups;
  auto makeGroup = [&](MemKind memory, int64_t id, ArrayRef<Operation *> allocs, bool circular = false,
                       bool mixedDepthPhysicalAlias = false) {
    groups.emplace_back();
    GroupDag &g = groups.back();
    g.bufferId = id;
    g.synthetic = syntheticIds.contains(id);
    g.mixedDepthPhysicalAlias = mixedDepthPhysicalAlias;
    g.memory = memory;
    g.circular = circular;
    for (Operation *allocOp : allocs) {
      Member m;
      m.allocOp = allocOp;
      m.type = cast<gpu::MemDescType>(allocOp->getResult(0).getType());
      m.circularStart = getI64Attr(allocOp, kBufferStartAttrName).value_or(0);
      m.offset = circular ? 0 : getI64Attr(allocOp, kBufferOffsetAttrName).value_or(0);
      m.extent = memory == MemKind::Tmem ? static_cast<int64_t>(mlir::triton::getMemDescSize(m.type))
                     : (m.type.getShape().empty() ? 1 : m.type.getShape().front());
      MemberId idx = static_cast<MemberId>(g.pieceTable.members.size());
      g.pieceTable.members.push_back(m);
      g.aliases.try_emplace(allocOp->getResult(0), std::make_pair(idx, SmallVector<AliasStep, 2>()));
    }
  };
  for (auto &[id, allocs] : tmemBuckets) {
    std::optional<int64_t> firstCopy;
    bool allAuthored = true;
    bool mixedCopies = false;
    for (Operation *alloc : allocs) {
      std::optional<int64_t> copy = getI64Attr(alloc, kBufferCopyAttrName);
      if (!copy) {
        allAuthored = false;
        break;
      }
      if (!firstCopy)
        firstCopy = *copy;
      else if (*firstCopy != *copy)
        mixedCopies = true;
    }
    if (!allAuthored || !mixedCopies) {
      makeGroup(MemKind::Tmem, id, allocs);
      continue;
    }
    for (Operation *alloc : allocs) {
      SmallVector<Operation *, 1> logicalMember{alloc};
      makeGroup(MemKind::Tmem, id, logicalMember, /*circular=*/false, /*mixedDepthPhysicalAlias=*/true);
    }
  }
  for (auto &[id, allocs] : localBuckets)
    makeGroup(MemKind::Local, id, allocs);
  for (Operation *allocOp : circularLocals)
    makeGroup(MemKind::Local, *getI64Attr(allocOp, kBufferIdAttrName),
              ArrayRef<Operation *>(allocOp), /*circular=*/true);
  return groups;
}

static void buildPieces(PieceTable &pt) {
  SmallVector<int64_t, 8> cuts;
  for (const Member &m : pt.members) {
    cuts.push_back(m.offset);
    cuts.push_back(m.offset + m.extent);
  }
  llvm::sort(cuts);
  cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
  SmallVector<Piece, 4> raw;
  for (size_t i = 0; i + 1 < cuts.size(); ++i) {
    Piece p;
    p.lo = cuts[i];
    p.hi = cuts[i + 1];
    for (auto [mIdx, m] : llvm::enumerate(pt.members))
      if (m.offset <= p.lo && p.hi <= m.offset + m.extent)
        p.cover.push_back(static_cast<MemberId>(mIdx));
    if (!p.cover.empty())
      raw.push_back(std::move(p));
  }
  for (Piece &p : raw) {
    if (!pt.pieces.empty() && pt.pieces.back().hi == p.lo && pt.pieces.back().cover == p.cover) {
      pt.pieces.back().hi = p.hi;
      continue;
    }
    pt.pieces.push_back(std::move(p));
  }
  pt.footprint.assign(pt.members.size(), {});
  for (auto [pIdx, piece] : llvm::enumerate(pt.pieces))
    for (MemberId m : piece.cover)
      pt.footprint[m].push_back(static_cast<PieceId>(pIdx));
  SmallVector<unsigned, 4> parent(pt.pieces.size());
  for (auto [i, _] : llvm::enumerate(parent))
    parent[i] = i;
  std::function<unsigned(unsigned)> find = [&](unsigned x) -> unsigned {
    while (parent[x] != x)
      x = parent[x] = parent[parent[x]];
    return x;
  };
  for (const auto &fp : pt.footprint)
    for (size_t i = 1; i < fp.size(); ++i)
      parent[find(fp[i])] = find(fp[0]);
  pt.pieceComp.assign(pt.pieces.size(), 0);
  DenseMap<unsigned, CompId> compId; // lookup-only
  CompId next = 0;
  for (auto [pIdx, _] : llvm::enumerate(pt.pieces)) {
    unsigned rep = find(static_cast<unsigned>(pIdx));
    auto it = compId.find(rep);
    if (it == compId.end())
      it = compId.try_emplace(rep, next++).first;
    pt.pieceComp[pIdx] = it->second;
  }
}
static bool isStructuralOp(Operation *op) {
  return isa<scf::ForOp, scf::IfOp, scf::YieldOp, triton::FuncOp, triton::ReturnOp>(op);
}
static LogicalResult rejectAliasOperands(GroupDag &g, Operation *op) {
  for (Value operand : op->getOperands())
    if (g.aliases.contains(operand))
      return semaError(op) << "unsupported memdesc flow through control-flow op " << op->getName();
  return success();
}
static FailureOr<bool> tryExtendAlias(GroupDag &g, Operation *op) {
  if (op->getNumResults() != 1 || !isa<gpu::MemDescType>(op->getResult(0).getType()))
    return false;
  for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
    auto it = g.aliases.find(operand);
    if (it == g.aliases.end())
      continue;
    if (!isSupportedAliasOp(op))
      return semaError(op) << "unsupported memdesc alias use " << op->getName();
    auto chain = it->second; // copy {member, steps}
    chain.second.push_back({op, static_cast<unsigned>(idx), op->getResult(0).getType()});
    g.aliases.try_emplace(op->getResult(0), std::move(chain));
    return true;
  }
  return false;
}
static void addTouch(GroupDag &g, SmallVectorImpl<Touch> &touches, Value v, Effect effect) {
  auto it = g.aliases.find(v);
  if (it == g.aliases.end())
    return;
  Touch t;
  t.member = it->second.first;
  t.effect = effect;
  t.accessValue = v;
  t.accessType = v.getType();
  t.alias = it->second.second;
  touches.push_back(std::move(t));
}
static LogicalResult collectTouches(GroupDag &g, Operation *op, SmallVectorImpl<Touch> &touches) {
  auto touch = [&](Value value, Effect effect) {
    addTouch(g, touches, value, effect);
    return success();
  };
  if (auto tmemAlloc = dyn_cast<nvidia_gpu::TMEMAllocOp>(op)) {
    if (tmemAlloc.getSrc())
      addTouch(g, touches, tmemAlloc.getResult(), Effect::W);
    return success();
  }
  if (auto localAlloc = dyn_cast<gpu::LocalAllocOp>(op)) {
    if (Value src = localAlloc.getSrc()) {
      if (Operation *def = src.getDefiningOp())
        if (isa<triton::DescriptorLoadOp, triton::DescriptorGatherOp>(def) &&
            g.aliases.count(localAlloc.getResult())) // member of THIS group
          g.ttDescriptorFedMembers.push_back(localAlloc);
      addTouch(g, touches, localAlloc.getResult(), Effect::W);
    }
    return success();
  }
  Value read, write;
  if (auto x = dyn_cast<nvidia_gpu::TMEMLoadOp>(op))
    read = x.getSrc();
  else if (auto x = dyn_cast<gpu::LocalLoadOp>(op))
    read = x.getSrc();
  else if (auto x = dyn_cast<nvidia_gpu::TMEMStoreOp>(op))
    write = x.getDst();
  else if (auto x = dyn_cast<gpu::LocalStoreOp>(op))
    write = x.getDst();
  else if (auto x = dyn_cast<nvws::DescriptorLoadOp>(op))
    write = x.getResult();
  else if (auto x = dyn_cast<nvws::DescriptorGatherOp>(op))
    write = x.getResult();
  if (read || write)
    return touch(read ? read : write, read ? Effect::R : Effect::W);
  if (auto mma = dyn_cast<nvidia_gpu::MMAv5OpInterface>(op)) {
    Value acc = mma.getAccumulator();
    bool accTouched = false;
    for (Value operand : op->getOperands()) {
      if (operand == acc) {
        if (!accTouched)
          addTouch(g, touches, operand, Effect::W);
        accTouched = true;
        continue;
      }
      addTouch(g, touches, operand, Effect::R);
    }
    return success();
  }
  if (isStructuralOp(op)) {
    return rejectAliasOperands(g, op);
  }
  for (Value operand : op->getOperands())
    if (g.aliases.contains(operand))
      addTouch(g, touches, operand, Effect::W);
  return success();
}

static FailureOr<Node *> buildChainForBlock(GroupDag &g, Block &block, Node *parent);
static void appendNode(Node *parent, Node *&head, Node *&tail, Node *n) {
  n->parent = parent;
  n->prev = tail;
  if (tail)
    tail->next = n;
  else
    head = n;
  tail = n;
}
static SmallVector<Operation *> directUsers(Value value) {
  SmallVector<Operation *> users;
  users.append(value.getUsers().begin(), value.getUsers().end());
  return users;
}

static LogicalResult deriveCompletionAnchor(Node *access) {
  auto load = dyn_cast<gpu::LocalLoadOp>(access->op);
  if (!load)
    return success();
  struct Candidate {
    Operation *forward = nullptr;
    Operation *store = nullptr;
  };
  SmallVector<Candidate, 2> candidates;
  for (Operation *user : load.getResult().getUsers()) {
    if (isa<triton::DescriptorStoreOp>(user)) {
      candidates.push_back({nullptr, user});
      continue;
    }
    auto convert = dyn_cast<gpu::ConvertLayoutOp>(user);
    if (!convert)
      continue;
    for (Operation *convertUser : convert.getResult().getUsers())
      if (isa<triton::DescriptorStoreOp>(convertUser))
        candidates.push_back({user, convertUser});
  }
  if (candidates.empty())
    return success();
  if (candidates.size() != 1)
    return semaError(load) << "managed local_load reaches multiple descriptor "
                              "stores; ownership completion is ambiguous";
  Candidate candidate = candidates.front();
  SmallVector<Operation *> loadUsers = directUsers(load.getResult());
  Operation *expectedLoadUser = candidate.forward ? candidate.forward : candidate.store;
  if (loadUsers.size() != 1 || loadUsers.front() != expectedLoadUser)
    return semaError(load) << "descriptor-store local_load path has fan-out";
  if (candidate.forward) {
    auto convert = cast<gpu::ConvertLayoutOp>(candidate.forward);
    SmallVector<Operation *> convertUsers = directUsers(convert.getResult());
    if (convertUsers.size() != 1 || convertUsers.front() != candidate.store)
      return semaError(load) << "descriptor-store convert_layout path has fan-out";
  }
  Block *block = load->getBlock();
  if (candidate.store->getBlock() != block || (candidate.forward && candidate.forward->getBlock() != block)) {
    InFlightDiagnostic diag = semaError(load) << "descriptor-store completion crosses control flow";
    diag.attachNote(candidate.store->getLoc()) << "descriptor store is here";
    return failure();
  }
  if (!load->isBeforeInBlock(candidate.store))
    return semaError(load) << "descriptor store must follow managed local_load";
  if (!sameOwner(access->owner, resolveOwner(candidate.store))) {
    InFlightDiagnostic diag = semaError(load) << "descriptor-store completion owner differs "
                                 "from managed local_load owner";
    diag.attachNote(candidate.store->getLoc()) << "descriptor store is here";
    return failure();
  }
  access->completionAnchor = candidate.store;
  return success();
}

static FailureOr<Node *> buildChainForBlock(GroupDag &g, Block &block, Node *parent) {
  Node *head = nullptr, *tail = nullptr;
  for (Operation &op : block) {
    if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
      if (failed(rejectAliasOperands(g, &op)))
        return failure();
      Node *forNode = g.newNode(Node::For, &op, parent);
      auto body = buildChainForBlock(g, *forOp.getBody(), forNode);
      if (failed(body))
        return failure();
      if (!*body) {
        g.nodes.pop_back(); // empty subtree: no structural node
        continue;
      }
      forNode->children.push_back(*body);
      appendNode(parent, head, tail, forNode);
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
      if (failed(rejectAliasOperands(g, &op)))
        return failure();
      Node *ifNode = g.newNode(Node::If, &op, parent);
      auto thenChain = buildChainForBlock(g, *ifOp.thenBlock(), ifNode);
      if (failed(thenChain))
        return failure();
      FailureOr<Node *> elseChain((Node *)nullptr);
      if (ifOp.elseBlock()) {
        elseChain = buildChainForBlock(g, *ifOp.elseBlock(), ifNode);
        if (failed(elseChain))
          return failure();
      }
      if (!*thenChain && !*elseChain) {
        g.nodes.pop_back();
        continue;
      }
      ifNode->children.push_back(*thenChain); // may be null
      ifNode->children.push_back(*elseChain); // may be null
      appendNode(parent, head, tail, ifNode);
      continue;
    }
    auto aliased = tryExtendAlias(g, &op);
    if (failed(aliased))
      return failure();
    if (*aliased)
      continue;
    SmallVector<Touch, 2> touches;
    if (failed(collectTouches(g, &op, touches)))
      return failure();
    if (touches.empty())
      continue;
    Node *access = g.newNode(Node::Access, &op, parent);
    access->owner = resolveOwner(&op);
    access->touches = std::move(touches);
    if (failed(deriveCompletionAnchor(access)))
      return failure();
    appendNode(parent, head, tail, access);
  }
  return head;
}
static void computeEffectSummary(GroupDag &g, Node *n, DenseMap<PieceId, Effect> &out) {
  if (n->kind == Node::Access) {
    forEachTouchedPiece(g, n, [&](PieceId p, Effect e) { mergeEffect(out, p, e); });
    return;
  }
  DenseMap<PieceId, Effect> sub;
  for (Node *childHead : n->children)
    for (Node *c = childHead; c; c = c->next)
      computeEffectSummary(g, c, sub);
  if (n->isRegion())
    for (const auto &[p, e] : sub)
      n->pieceInfo[p] = PieceInfo{std::nullopt, e};
  for (const auto &[p, e] : sub)
    mergeEffect(out, p, e);
}

LogicalResult buildAccessDag(GroupDag &g, triton::FuncOp funcOp) {
  buildPieces(g.pieceTable);
  Node *func = g.newNode(Node::Func, funcOp, nullptr);
  auto chain = buildChainForBlock(g, funcOp.getBody().front(), func);
  if (failed(chain))
    return failure();
  if (*chain)
    func->children.push_back(*chain);
  g.root = func;
  DenseMap<PieceId, Effect> ignored;
  computeEffectSummary(g, func, ignored);
  return success();
}

void dumpGroupAccessDag(GroupDag &g, triton::FuncOp funcOp) {
  auto &os = llvm::errs();
  os << "GROUP ";
  if (g.synthetic)
    os << "buffer.id=none#" << -g.bufferId;
  else
    os << "buffer.id=" << g.bufferId;
  os << " memory=" << (g.isTmem() ? "tmem" : "local") << " members=" << g.pieceTable.members.size() << "\n";
  os << "  members:";
  for (auto [idx, m] : llvm::enumerate(g.pieceTable.members))
    os << " m" << idx << "[" << m.offset << "," << (m.offset + m.extent) << ")";
  os << "\n  pieces:";
  for (auto [idx, p] : llvm::enumerate(g.pieceTable.pieces)) {
    os << " P" << idx << "=[" << p.lo << "," << p.hi << "){";
    llvm::interleaveComma(p.cover, os, [&](MemberId m) { os << "m" << m; });
    os << "}c" << g.pieceTable.pieceComp[idx];
  }
  os << "\n  footprints:";
  for (auto [idx, fp] : llvm::enumerate(g.pieceTable.footprint)) {
    os << " m" << idx << "={";
    llvm::interleaveComma(fp, os, [&](PieceId p) { os << "P" << p; });
    os << "}";
  }
  os << "\nACCESS-DAG\n";
  os << "|- func @" << funcOp.getName() << "\n";
  dumpDagTree(g, DumpStage::Access);
}
} // namespace mlir::triton::nvws_semas
