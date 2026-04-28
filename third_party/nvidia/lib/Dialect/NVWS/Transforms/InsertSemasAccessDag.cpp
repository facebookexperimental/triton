// ACCESS and boundary-owner analysis; see sema-docs/insert-semas/access-dag.md.
#include "InsertSemas.h"

namespace mlir::triton::nvws_semas {

FailureOr<SmallVector<GroupDag, 0>> collectGroups(triton::FuncOp funcOp,
                                                  Block *functionBlock) {
  using Buckets = llvm::MapVector<int64_t, SmallVector<Operation *, 2>>;
  Buckets tmemBuckets, localBuckets;
  SmallVector<Operation *, 4> circularLocals;
  int64_t nextSynthetic = -1;
  auto add = [&](Buckets &buckets, Operation *op, std::optional<int64_t> id) {
    int64_t key = id ? *id : nextSynthetic--;
    buckets[key].push_back(op);
  };
  LogicalResult result = success();
  auto collect = [&](Operation *op) {
    std::optional<int64_t> id = getI64Attr(op, kBufferIdAttrName);
    if (isa<nvidia_gpu::TMEMAllocOp>(op)) {
      add(tmemBuckets, op, id);
      return;
    }
    auto alloc = dyn_cast<gpu::LocalAllocOp>(op);
    if (!alloc || !cast<gpu::MemDescType>(alloc.getType()).getMutableMemory())
      return;
    if (!op->hasAttr(kBufferCircularAttrName)) {
      add(localBuckets, op, id);
      return;
    }
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
  };
  if (functionBlock) {
    for (Operation &op : *functionBlock)
      op.walk(collect);
  } else {
    funcOp.walk(collect);
  }
  if (failed(result))
    return failure();

  SmallVector<GroupDag, 0> groups;
  auto makeGroup = [&](MemKind memory, int64_t id, ArrayRef<Operation *> allocs,
                       bool circular = false) {
    GroupDag &g = groups.emplace_back();
    g.bufferId = id;
    g.memory = memory;
    g.circular = circular;
    for (Operation *op : allocs) {
      auto type = cast<gpu::MemDescType>(op->getResult(0).getType());
      int64_t extent = memory == MemKind::Tmem
                           ? static_cast<int64_t>(mlir::triton::getMemDescSize(type))
                           : (type.getShape().empty() ? 1 : type.getShape().front());
      Member member{op, type,
                    circular ? 0 : getI64Attr(op, kBufferOffsetAttrName).value_or(0),
                    extent,
                    getI64Attr(op, kBufferCopyAttrName).value_or(1),
                    getI64Attr(op, kBufferStartAttrName).value_or(0)};
      MemberId index = g.pieceTable.members.size();
      g.pieceTable.members.push_back(member);
      g.aliases.try_emplace(op->getResult(0),
                            std::make_pair(index, SmallVector<AliasStep, 2>()));
    }
  };
  for (auto &[id, allocs] : tmemBuckets) {
    std::optional<int64_t> expectedCopy;
    Operation *expectedCopyOp = nullptr;
    for (Operation *op : allocs) {
      std::optional<int64_t> copy = getI64Attr(op, kBufferCopyAttrName);
      if (!copy)
        continue;
      if (!expectedCopy) {
        expectedCopy = copy;
        expectedCopyOp = op;
        continue;
      }
      if (copy == expectedCopy)
        continue;
      InFlightDiagnostic diag = semaError(op)
                                << "TMEM allocations sharing buffer.id " << id
                                << " have conflicting buffer.copy values "
                                << *expectedCopy << " and " << *copy;
      diag.attachNote(expectedCopyOp->getLoc())
          << "first buffer.copy value is " << *expectedCopy;
      return failure();
    }
    makeGroup(MemKind::Tmem, id, allocs);
  }
  for (auto &[id, allocs] : localBuckets)
    makeGroup(MemKind::Local, id, allocs);
  for (Operation *op : circularLocals)
    makeGroup(MemKind::Local, *getI64Attr(op, kBufferIdAttrName),
              ArrayRef<Operation *>(op), true);
  return groups;
}

static bool buildPieces(PieceTable &pt) {
  SmallVector<int64_t, 8> cuts;
  for (const Member &member : pt.members) {
    cuts.push_back(member.offset);
    cuts.push_back(member.offset + member.extent);
  }
  llvm::sort(cuts);
  cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
  pt.footprint.assign(pt.members.size(), {});
  SmallVector<MemberId, 2> previous;
  PieceId piece = 0;
  for (size_t i = 0; i + 1 < cuts.size(); ++i) {
    SmallVector<MemberId, 2> cover;
    for (auto [index, member] : llvm::enumerate(pt.members))
      if (member.offset <= cuts[i] && cuts[i + 1] <= member.offset + member.extent)
        cover.push_back(static_cast<MemberId>(index));
    if (cover.empty()) return false;
    if (previous == cover) continue;
    if (!previous.empty() && llvm::none_of(previous, [&](MemberId member) {
          return llvm::is_contained(cover, member);
        }))
      return false;
    for (MemberId member : cover)
      pt.footprint[member].push_back(piece);
    previous = std::move(cover);
    ++piece;
  }
  return true;
}
static LogicalResult rejectAliasOperands(GroupDag &g, Operation *op) {
  for (Value operand : op->getOperands())
    if (g.aliases.contains(operand))
      return semaError(op) << "unsupported memdesc flow through control-flow op " << op->getName();
  return success();
}
static LogicalResult collectTouches(GroupDag &g, Operation *op, SmallVectorImpl<Touch> &touches) {
  if (op->getNumResults() == 1 && isa<gpu::MemDescType>(op->getResult(0).getType()))
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      auto it = g.aliases.find(operand);
      if (it == g.aliases.end())
        continue;
      if (!isSupportedAliasOp(op))
        return semaError(op) << "unsupported memdesc alias use " << op->getName();
      auto alias = it->second;
      alias.second.push_back(
          {op, static_cast<unsigned>(index), op->getResult(0).getType()});
      g.aliases.try_emplace(op->getResult(0), std::move(alias));
      return success();
    }
  auto touch = [&](Value value, Effect effect) {
    auto it = g.aliases.find(value);
    if (it != g.aliases.end())
      touches.push_back(Touch{it->second.first, effect, value, value.getType(),
                              it->second.second});
  };
  if (auto tmemAlloc = dyn_cast<nvidia_gpu::TMEMAllocOp>(op)) {
    if (tmemAlloc.getSrc())
      touch(tmemAlloc.getResult(), Effect::W);
    return success();
  }
  if (auto localAlloc = dyn_cast<gpu::LocalAllocOp>(op)) {
    if (Value src = localAlloc.getSrc();
        src && g.aliases.contains(localAlloc.getResult())) {
      if (isa_and_nonnull<triton::DescriptorLoadOp,
                          triton::DescriptorGatherOp>(src.getDefiningOp()))
        g.ttDescriptorFedMembers.push_back(localAlloc);
      touch(localAlloc.getResult(), Effect::W);
    }
    return success();
  }
  Value read, write;
  if (auto x = dyn_cast<nvidia_gpu::TMEMLoadOp>(op))
    read = x.getSrc();
  else if (auto x = dyn_cast<gpu::LocalLoadOp>(op))
    read = x.getSrc();
  else if (auto x =
               dyn_cast<nvidia_gpu::AsyncTMACopyLocalToGlobalOp>(op))
    read = x.getSrc();
  else if (auto x = dyn_cast<nvidia_gpu::AsyncTMAReduceOp>(op))
    read = x.getSrc();
  else if (auto x = dyn_cast<nvidia_gpu::TMEMStoreOp>(op))
    write = x.getDst();
  else if (auto x = dyn_cast<gpu::LocalStoreOp>(op))
    write = x.getDst();
  else if (auto x = dyn_cast<nvws::DescriptorLoadOp>(op))
    write = x.getResult();
  else if (auto x = dyn_cast<nvws::DescriptorGatherOp>(op))
    write = x.getResult();
  if (read || write) {
    touch(read ? read : write, read ? Effect::R : Effect::W);
    return success();
  }
  if (auto mma = dyn_cast<nvidia_gpu::MMAv5OpInterface>(op)) {
    Value acc = mma.getAccumulator();
    touch(acc, Effect::W);
    for (Value operand : op->getOperands())
      if (operand != acc) touch(operand, Effect::R);
    return success();
  }
  if (isa<scf::YieldOp, triton::FuncOp, triton::ReturnOp>(op))
    return rejectAliasOperands(g, op);
  for (Value operand : op->getOperands())
    if (g.aliases.contains(operand))
      touch(operand, Effect::W);
  return success();
}

// Transient construction summary.  The persistent graph remains Node-only;
// these facts let ACCESS assign region owners and boundary records while it
// builds that graph instead of recovering them in a second analysis pass.
struct Chain {
  Node *head = nullptr, *tail = nullptr;
  DenseMap<PieceId, Effect> effects;
  DenseMap<PieceId, Owner> firstOwners, lastOwners;
};
static void appendNode(GroupDag &g, Chain &chain, Node *node) {
  node->prev = chain.tail;
  if (chain.tail)
    chain.tail->next = node;
  else
    chain.head = node;
  chain.tail = node;
  node->slotEffect = Effect::R;
  auto record = [&](PieceId piece, Effect effect, const Owner &owner) {
    node->slotEffect = joinEffect(*node->slotEffect, effect);
    mergeEffect(chain.effects, piece, effect);
    chain.firstOwners.try_emplace(piece, owner);
    chain.lastOwners[piece] = owner;
  };
  if (node->kind == Node::Access) {
    forEachTouchedPiece(g, node, [&](PieceId piece, Effect effect) {
      record(piece, effect, node->owner);
    });
  } else {
    assert(node->isRegion() && "appendNode expects an access or region");
    bool sealed = node->kind == Node::For && gpu::hasWarpSpecializeTag(node->op);
    for (auto [piece, info] : node->pieceInfo)
      record(piece, info.effect, sealed ? Owner() : info.owner);
  }
}

// Work around early TMA-store lowering exposing ttng TMA operations directly
// to InsertSemas. Their SMEM source remains live until the token wait, not just
// until the asynchronous operation is issued. This special handling should be
// replaced by introducing buffer-taking `nvws.descriptor_store` and
// `nvws.descriptor_reduce` counterparts and retaining them through
// InsertSemas, as described in the NVWS transition documentation.
static LogicalResult deriveTMACompletionAnchor(Node *access, Value token) {
  Operation *tma = access->op;
  if (!token)
    return semaError(tma)
           << "managed TMA store requires an async completion token";
  if (!llvm::hasSingleElement(token.getUsers()))
    return semaError(tma)
           << "managed TMA store token must have exactly one user";
  auto wait = dyn_cast<nvidia_gpu::TMAStoreTokenWaitOp>(
      *token.getUsers().begin());
  if (!wait)
    return semaError(tma)
           << "managed TMA store token user must be "
              "ttng.async_tma_store_token_wait";
  if (wait.getToken() != token)
    return semaError(tma)
           << "managed TMA store token must be the completion token of "
              "ttng.async_tma_store_token_wait";
  if (wait->getBlock() != tma->getBlock()) {
    InFlightDiagnostic diag =
        semaError(tma) << "managed TMA store completion crosses control flow";
    diag.attachNote(wait.getLoc()) << "completion wait is here";
    return failure();
  }
  if (!tma->isBeforeInBlock(wait))
    return semaError(tma)
           << "managed TMA store completion wait must follow the TMA store";
  if (!sameOwner(access->owner, resolveOwner(wait))) {
    InFlightDiagnostic diag =
        semaError(tma) << "managed TMA store completion owner differs from "
                          "the TMA store owner";
    diag.attachNote(wait.getLoc()) << "completion wait is here";
    return failure();
  }
  access->completionAnchor = wait;
  return success();
}

static LogicalResult deriveCompletionAnchor(Node *access) {
  if (auto copy =
          dyn_cast<nvidia_gpu::AsyncTMACopyLocalToGlobalOp>(access->op))
    return deriveTMACompletionAnchor(access, copy.getToken());
  if (auto reduce = dyn_cast<nvidia_gpu::AsyncTMAReduceOp>(access->op))
    return deriveTMACompletionAnchor(access, reduce.getToken());

  auto load = dyn_cast<gpu::LocalLoadOp>(access->op);
  if (!load)
    return success();
  Operation *forward = nullptr, *store = nullptr;
  for (Operation *user : load.getResult().getUsers()) {
    if (isa<triton::DescriptorStoreOp>(user)) {
      forward = nullptr;
      store = user;
      continue;
    }
    if (auto convert = dyn_cast<gpu::ConvertLayoutOp>(user))
      for (Operation *convertUser : convert.getResult().getUsers())
        if (isa<triton::DescriptorStoreOp>(convertUser)) {
          forward = user;
          store = convertUser;
        }
  }
  if (!store)
    return success();
  auto onlyUser = [](Value value, Operation *expected) {
    return llvm::hasSingleElement(value.getUsers()) && *value.getUsers().begin() == expected;
  };
  if (!onlyUser(load.getResult(), forward ? forward : store))
    return semaError(load) << "descriptor-store local_load path has fan-out";
  if (forward && !onlyUser(forward->getResult(0), store))
    return semaError(load) << "descriptor-store convert_layout path has fan-out";
  Block *block = load->getBlock();
  if (store->getBlock() != block || (forward && forward->getBlock() != block)) {
    InFlightDiagnostic diag = semaError(load) << "descriptor-store completion crosses control flow";
    diag.attachNote(store->getLoc()) << "descriptor store is here";
    return failure();
  }
  if (!load->isBeforeInBlock(store))
    return semaError(load) << "descriptor store must follow managed local_load";
  if (!sameOwner(access->owner, resolveOwner(store))) {
    InFlightDiagnostic diag = semaError(load) << "descriptor-store completion owner differs "
                                 "from managed local_load owner";
    diag.attachNote(store->getLoc()) << "descriptor store is here";
    return failure();
  }
  access->completionAnchor = store;
  return success();
}

static FailureOr<Chain> buildChainForBlock(GroupDag &g, Block &block, Node *parent) {
  Chain chain;
  for (Operation &op : block) {
    Node::Kind kind = isa<scf::ForOp>(op) ? Node::For
                      : isa<scf::IfOp>(op) ? Node::If
                                           : Node::Access;
    if (kind != Node::Access) {
      if (failed(rejectAliasOperands(g, &op)))
        return failure();
      Node *region = g.newNode(kind, &op, parent);
      SmallVector<Chain, 2> branches;
      for (Region &nested : op.getRegions()) {
        if (nested.empty()) {
          branches.emplace_back();
          continue;
        }
        auto branch = buildChainForBlock(g, nested.front(), region);
        if (failed(branch))
          return failure();
        branches.push_back(std::move(*branch));
      }
      assert((kind != Node::If || branches.size() == 2) &&
             "If node carries then+else slots");
      if (llvm::all_of(branches, [](const Chain &branch) { return !branch.head; })) {
        g.discardLastNode(region);
        continue;
      }
      for (const Chain &branch : branches)
        for (auto [piece, effect] : branch.effects) {
          auto [it, inserted] = region->pieceInfo.try_emplace(
              piece, PieceInfo{std::nullopt, effect});
          if (!inserted)
            it->second.effect = joinEffect(it->second.effect, effect);
        }
      auto lookup = [](const DenseMap<PieceId, Owner> &owners,
                       PieceId piece) -> const Owner * {
        auto it = owners.find(piece);
        return it == owners.end() ? nullptr : &it->second;
      };
      for (auto [piece, info] : sortedPieceInfo(region)) {
        const Owner *owner = region->kind == Node::If
                                 ? lookup(chain.lastOwners, piece)
                                 : nullptr;
        for (const Chain &branch : branches)
          if (!owner)
            owner = lookup(branch.firstOwners, piece);
        assert(owner && "piece summary must have a first toucher");
        region->pieceInfo[piece].owner = *owner;
      }
      for (const Chain &branch : branches) {
        Node *enter = g.newNode(Node::Enter, nullptr, region);
        Node *exit = g.newNode(Node::Exit, nullptr, region);
        for (auto [piece, effect] : branch.effects)
          enter->pieceInfo[piece] = exit->pieceInfo[piece] =
              PieceInfo{region->pieceInfo[piece].owner, effect};
        Node *head = branch.head ? branch.head : exit;
        Node *tail = branch.tail ? branch.tail : enter;
        enter->next = head;
        head->prev = enter;
        tail->next = exit;
        exit->prev = tail;
        region->children.push_back(enter);
      }
      appendNode(g, chain, region);
      continue;
    }
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
    g.accessNodeOps.insert(&op);
    appendNode(g, chain, access);
  }
  return chain;
}

LogicalResult buildAccessDag(GroupDag &g, triton::FuncOp funcOp,
                             Block &functionBlock) {
  // Single-component invariant: every buffer.id group's pieces connect
  // (through shared members) into one component -- the memory planner keeps
  // reusers stacked within their owner's columns. The rest of InsertSemas
  // relies on this (one group == one synchronization unit); reject anything
  // that violates it rather than mis-synchronizing.
  if (!buildPieces(g.pieceTable))
    return semaError(g.pieceTable.members.front().allocOp)
           << "buffer.id group has disjoint pieces (more than "
                            "one connected component); InsertSemas requires "
                            "one component per group";
  Node *func = g.newNode(Node::Func, funcOp, nullptr);
  auto chain = buildChainForBlock(g, functionBlock, func);
  if (failed(chain))
    return failure();
  if (chain->head)
    func->children.push_back(chain->head);
  g.root = func;
  return success();
}

} // namespace mlir::triton::nvws_semas
