// Protocol materialization; see sema-docs/insert-semas/emit-ir.md.
#include "InsertSemas.h"

namespace mlir::triton::nvws_semas {

struct EmitCtx {
  triton::FuncOp func;
  Value poison; // the single function-level ub.poison token (contract E)
  Type tokenType;
  struct Slot {
    GroupDag *g;
    CompId comp;
    unsigned index; // absolute result / iter_arg index in the NEW op
  };
  llvm::MapVector<Operation *, SmallVector<Slot, 2>> slots;
};

struct RenderState {
  DenseMap<CompId, Value> carrier;          // current carrier token per comp
  DenseMap<CompId, Value> carrierSema;      // the create of that carrier's
  DenseMap<MemberId, Value> view;           // member view cache
  void clearViews() { view.clear(); }
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
  auto op = gpu::createInto<OpT>(b, loc, ids, stageCluster, std::forward<Args>(args)...);
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
static Operation *nextRealOp(const Node *n) {
  for (const Node *m = n; m; m = m->next)
    if ((m->kind == Node::Access || m->kind == Node::For || m->kind == Node::If) && m->op)
      return m->op;
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
static void nukeGroupTokens(EmitCtx &ctx, GroupDag &g) {
  auto nukeOp = [&](Operation *op) {
    if (auto l = dyn_cast<nvidia_gpu::TMEMLoadOp>(op)) {
      l.getDepMutable().clear();
      if (l.getToken())
        l.getToken().replaceAllUsesWith(ctx.poison);
      return;
    }
    if (auto st = dyn_cast<nvidia_gpu::TMEMStoreOp>(op)) {
      st.getDepMutable().clear();
      if (st.getToken())
        st.getToken().replaceAllUsesWith(ctx.poison);
      return;
    }
    if (auto mma = dyn_cast<nvidia_gpu::MMAv5OpInterface>(op)) {
      if (g.aliases.count(mma.getAccumulator())) {
        mma.getAccDepMutable().clear();
        if (mma.getToken())
          mma.getToken().replaceAllUsesWith(ctx.poison);
      }
      return;
    }
    if (auto a = dyn_cast<nvidia_gpu::TMEMAllocOp>(op)) {
      if (a.getToken())
        a.getToken().replaceAllUsesWith(ctx.poison);
      return;
    }
  };
  for (const Member &m : g.pieceTable.members)
    nukeOp(m.allocOp);
  forEachNode(g, [&](Node *n) {
    if (n->kind == Node::Access)
      nukeOp(n->op);
  });
}
static bool isScalesEnc(gpu::MemDescType t) {
  return isa<nvidia_gpu::TensorMemoryScalesEncodingAttr>(t.getEncoding());
}
static gpu::MemDescType backingType(const GroupDag &g, const Member &m) {
  auto t = m.type;
  SmallVector<int64_t> shape(t.getShape());
  if (!isScalesEnc(t))
    shape.insert(shape.begin(), g.numStages);
  return gpu::MemDescType::get(shape, t.getElementType(), t.getEncoding(),
                               t.getMemorySpace(), /*mutableMemory=*/true);
}
static gpu::MemDescType withMutable(gpu::MemDescType t, bool m) {
  if (t.getMutableMemory() == m)
    return t;
  return gpu::MemDescType::get(t.getShape(), t.getElementType(),
                               t.getEncoding(), t.getMemorySpace(), m, t.getAllocShape());
}
static gpu::MemDescType genericViewType(gpu::MemDescType backing) {
  auto shape = backing.getShape();
  return gpu::MemDescType::get(isScalesEnc(backing) ? shape : shape.drop_front(),
                               backing.getElementType(), backing.getEncoding(), backing.getMemorySpace(),
                               /*mutableMemory=*/true, backing.getShape());
}
static bool sameViewType(Type a, Type b) {
  auto x = dyn_cast<gpu::MemDescType>(a), y = dyn_cast<gpu::MemDescType>(b);
  if (!x || !y)
    return false;
  return x.getShape() == y.getShape() && x.getElementType() == y.getElementType() &&
         x.getEncoding() == y.getEncoding() && x.getMemorySpace() == y.getMemorySpace() &&
         x.getMutableMemory() == y.getMutableMemory();
}
static gpu::MemDescType localViewType(const GroupDag &g, MemberId member,
                                      ArrayRef<const Touch *> touches, gpu::MemDescType backing) {
  for (const Touch *t : touches) {
    if (t->member != member)
      continue;
    Type ty = g.pieceTable.members[member].type;
    if (t->alias.empty())
      ty = t->accessType;
    for (const AliasStep &step : t->alias) {
      if (step.op->getName().getStringRef() != "ttg.memdesc_index")
        break;
      ty = step.resultType;
    }
    if (auto at = dyn_cast<gpu::MemDescType>(ty))
      return withMutable(at, true);
  }
  return withMutable(genericViewType(backing), true);
}

static void emitBackingsAndCreates(EmitCtx &ctx, GroupDag &g) {
  if (g.semas.empty())
    return;
  OpBuilder b(ctx.func);
  Operation *anchor = g.pieceTable.members.front().allocOp;
  for (const Member &m : g.pieceTable.members)
    if (m.allocOp->getBlock() == anchor->getBlock() && m.allocOp->isBeforeInBlock(anchor))
      anchor = m.allocOp;
  while (isa<scf::ForOp>(anchor->getParentOp()))
    anchor = anchor->getParentOp();
  b.setInsertionPoint(anchor);
  Location loc = anchor->getLoc();
  g.backing.clear();
  SmallVector<Type> baseTypes;
  for (const Member &m : g.pieceTable.members) {
    auto bt = backingType(g, m);
    Value backing;
    if (g.isTmem()) {
      auto alloc = nvidia_gpu::TMEMAllocOp::create(b, loc, bt, /*src=*/Value());
      backing = alloc.getResult();
    } else {
      auto alloc = gpu::LocalAllocOp::create(b, loc, bt);
      backing = alloc.getResult();
    }
    SmallVector<StringRef, 5> attrNames{
        kBufferIdAttrName, kBufferOffsetAttrName, kBufferCopyAttrName,
        kBufferCircularAttrName, kBufferStartAttrName};
    for (StringRef name : attrNames)
      if (Attribute a = m.allocOp->getAttr(name))
        backing.getDefiningOp()->setAttr(name, a);
    g.backing.push_back(backing);
    baseTypes.push_back(bt);
  }
  auto semaTy = nvws::SemaphoreType::get(
      b.getContext(), nvws::TypeArrayAttr::get(b.getContext(), baseTypes));
  SmallVector<Sema *, 4> order;
  for (Sema &s : g.semas)
    if (s.isEntry)
      order.push_back(&s);
  for (Sema &s : g.semas)
    if (!s.isEntry)
      order.push_back(&s);
  for (Sema *sp : order) {
    Sema &s = *sp;
    auto create = nvws::SemaphoreCreateOp::create(b, loc, semaTy, g.backing, s.isEntry);
    create.setPendingCountAttr(b.getI32IntegerAttr(s.count));
    s.create = create.getResult();
  }
}
static void emitEntryAcquires(EmitCtx &ctx, GroupDag &g, DenseMap<Node *, Value> &emitted) {
  forEachNode(g, [&](Node *n) {
    if (n->kind != Node::Acquire || n->owner || !getSema(g, n).isEntry || emitted.count(n))
      return;
    Operation *before = nextRealOp(n->next);
    OpBuilder b(ctx.func);
    if (before)
      b.setInsertionPoint(before);
    else
      b.setInsertionPoint(n->parent && n->parent->op ? n->parent->op->getRegion(0).front().getTerminator()
                              : &ctx.func.getBody().front().back());
    const Sema &s = getSema(g, n);
    auto acq = emitInto<nvws::SemaphoreAcquireOp>(
        b, before ? before->getLoc() : ctx.func.getLoc(), Owner(), {}, s.create, ctx.tokenType);
    if (n->stageOffset)
      acq.setStage(materializeI32Before(acq, *n->stageOffset));
    emitted[n] = acq.getToken();
  });
}
static void fixupAnchors(MutableArrayRef<GroupDag> groups, Operation *oldOp, Operation *newOp) {
  for (GroupDag &g : groups) {
    forEachNode(g, [&](Node *n) {
      if (n->op == oldOp)
        n->op = newOp;
    });
  }
}
static void filterPartitionOutputs(Operation *op, ArrayRef<bool> keep) {
  auto attr = op->getAttrOfType<ArrayAttr>(gpu::kPartitionOutputsAttrName);
  if (!attr || attr.size() != keep.size())
    return;
  SmallVector<Attribute> kept;
  for (auto [a, k] : llvm::zip(attr.getValue(), keep))
    if (k)
      kept.push_back(a);
  op->setAttr(gpu::kPartitionOutputsAttrName, ArrayAttr::get(op->getContext(), kept));
}
static void finishResultFilter(Operation *oldOp, Operation *newOp, ArrayRef<bool> keep,
                               MutableArrayRef<GroupDag> groups) {
  filterPartitionOutputs(newOp, keep);
  unsigned next = 0;
  for (auto [i, result] : llvm::enumerate(oldOp->getResults()))
    if (keep[i])
      result.replaceAllUsesWith(newOp->getResult(next++));
  fixupAnchors(groups, oldOp, newOp);
  oldOp->erase();
}
static void eraseDroppedYields(Operation *op, ArrayRef<bool> keep) {
  for (Region &region : op->getRegions()) {
    if (region.empty())
      continue;
    Operation *yield = region.front().getTerminator();
    for (int i = keep.size() - 1; i >= 0; --i)
      if (!keep[i])
        yield->eraseOperand(i);
  }
}

static bool eraseDeadTokenSlots(EmitCtx &ctx, MutableArrayRef<GroupDag> groups) {
  bool changed = false;
  SmallVector<scf::ForOp> fors;
  SmallVector<scf::IfOp> ifs;
  ctx.func.walk([&](Operation *op) {
    if (auto f = dyn_cast<scf::ForOp>(op))
      fors.push_back(f);
    else if (auto i = dyn_cast<scf::IfOp>(op))
      ifs.push_back(i);
  });
  for (scf::IfOp ifOp : ifs) {
    SmallVector<bool> keep(ifOp.getNumResults(), true);
    bool any = false;
    for (auto [i, res] : llvm::enumerate(ifOp.getResults()))
      if (res.getType() == ctx.tokenType && res.use_empty()) {
        keep[i] = false;
        any = true;
      }
    if (!any)
      continue;
    SmallVector<Type> keptTypes;
    for (auto [i, res] : llvm::enumerate(ifOp.getResults()))
      if (keep[i])
        keptTypes.push_back(res.getType());
    OpBuilder b(ifOp);
    auto newIf = scf::IfOp::create(b, ifOp.getLoc(), keptTypes, ifOp.getCondition(),
                                   /*withElseRegion=*/!ifOp.getElseRegion()
                                       .empty());
    newIf->setAttrs(ifOp->getAttrs());
    newIf.getThenRegion().takeBody(ifOp.getThenRegion());
    if (!ifOp.getElseRegion().empty())
      newIf.getElseRegion().takeBody(ifOp.getElseRegion());
    eraseDroppedYields(newIf, keep);
    finishResultFilter(ifOp, newIf, keep, groups);
    changed = true;
  }
  for (scf::ForOp forOp : fors) {
    SmallVector<bool> keep(forOp.getNumResults(), true);
    bool any = false;
    for (auto [i, res] : llvm::enumerate(forOp.getResults()))
      if (res.getType() == ctx.tokenType && res.use_empty() && forOp.getRegionIterArg(i).use_empty()) {
        keep[i] = false;
        any = true;
      }
    if (!any)
      continue;
    SmallVector<Value> keptInits;
    for (auto [i, init] : llvm::enumerate(forOp.getInits()))
      if (keep[i])
        keptInits.push_back(init);
    OpBuilder b(forOp);
    auto newFor = scf::ForOp::create(b, forOp.getLoc(), forOp.getLowerBound(),
                           forOp.getUpperBound(), forOp.getStep(), keptInits);
    newFor->setAttrs(forOp->getAttrs());
    newFor.getRegion().takeBody(forOp.getRegion());
    Block &body = newFor.getRegion().front();
    for (int i = keep.size() - 1; i >= 0; --i)
      if (!keep[i]) {
        body.getTerminator()->eraseOperand(i);
        body.eraseArgument(1 + i); // +1: induction variable
      }
    finishResultFilter(forOp, newFor, keep, groups);
    changed = true;
  }
  return changed;
}

// Aggregate every group's token slots before rebuilding a region exactly once.
static void rewriteSignatures(EmitCtx &ctx, MutableArrayRef<GroupDag> groups) {
  struct Want {
    GroupDag *g;
    CompId comp;
    Owner owner;
  };
  llvm::MapVector<Operation *, SmallVector<Want, 2>> wanted;
  for (GroupDag &g : groups) {
    forEachNode(g, [&](Node *n) {
      if (n->isRegion())
        for (const Crossing &c : n->crossings) {
          if (n->kind == Node::For && !c.hold.materializesCarrier())
            continue;
          wanted[n->op].push_back(Want{&g, c.comp, c.slotOwner});
        }
    });
  }
  SmallVector<Operation *> ops;
  for (auto &[op, _] : wanted)
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
    auto &list = wanted[op];
    unsigned nSlots = list.size();
    SmallVector<Value> poisons(nSlots, ctx.poison);
    Operation *newOp = nullptr;
    unsigned base = 0;
    SmallVector<SetVector<int>, 4> outputs;
    if (op->getNumResults() == 0 || op->hasAttr(gpu::kPartitionOutputsAttrName))
      outputs = gpu::getPartitionOutputs(op);
    else
      outputs.assign(op->getNumResults(), partitionIdsOfFwd(op));
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      base = forOp.getNumRegionIterArgs();
      b.setInsertionPoint(forOp);
      auto newLoop = addIterArgsToLoop(b, forOp, poisons);
      appendToForOpYield(newLoop, poisons);
      newOp = newLoop;
    } else {
      auto ifOp = cast<scf::IfOp>(op);
      base = ifOp.getNumResults();
      SmallVector<Type> types(nSlots, ctx.tokenType);
      b.setInsertionPoint(ifOp);
      auto newIf = replaceIfOpWithNewSignature(b, ifOp, types);
      for (Block *blk : {newIf.thenBlock(), newIf.elseBlock()}) {
        auto yield = cast<scf::YieldOp>(blk->getTerminator());
        if (yield.getNumOperands() < newIf.getNumResults()) {
          SmallVector<Value> operands(yield.getOperands());
          operands.append(newIf.getNumResults() - yield.getNumOperands(), ctx.poison);
          OpBuilder yb(yield);
          auto ny = scf::YieldOp::create(yb, yield.getLoc(), operands);
          ny->setAttrs(yield->getAttrs());
          yield.erase();
        }
      }
      ifOp.erase(); // the husk (utility does not erase)
      newOp = newIf;
    }
    for (const Want &w : list) {
      SetVector<int> set;
      if (w.owner)
        set.insert(w.owner->first);
      else
        set = partitionIdsOfFwd(newOp);
      outputs.push_back(set);
    }
    if (gpu::hasPartition(newOp))
      gpu::setPartitionOutputs(newOp, outputs);
    if (gpu::hasPartition(newOp)) {
      auto ids = gpu::getPartitionIds(newOp);
      for (Region &r : newOp->getRegions())
        for (Block &blk : r)
          if (Operation *term = blk.getTerminator())
            if (!gpu::hasPartition(term))
              term->setAttr(gpu::kPartitionAttrName, DenseI32ArrayAttr::get(newOp->getContext(),
                      SmallVector<int>(ids.begin(), ids.end())));
    }
    fixupAnchors(groups, op, newOp);
    auto &rec = ctx.slots[newOp];
    for (auto [i, w] : llvm::enumerate(list))
      rec.push_back(EmitCtx::Slot{w.g, w.comp, base + static_cast<unsigned>(i)});
  }
}
static unsigned slotIndexFor(EmitCtx &ctx, Operation *op, GroupDag *g, CompId comp) {
  for (const EmitCtx::Slot &s : ctx.slots[op])
    if (s.g == g && s.comp == comp)
      return s.index;
  llvm_unreachable("missing slot");
}
static Value getView(GroupDag &g, RenderState &rs, Node *node, const Touch &t, Operation *accessOp,
                     const Owner &owner) {
  auto it = rs.view.find(t.member);
  Value base;
  if (it != rs.view.end()) {
    base = it->second;
  } else {
    OpBuilder b(accessOp);
    SmallVector<Type> types;
    for (auto [mi, m] : llvm::enumerate(g.pieceTable.members)) {
      auto bt = cast<gpu::MemDescType>(g.backing[mi].getType());
      if (g.isTmem())
        types.push_back(genericViewType(bt));
      else
        types.push_back(localViewType(g, static_cast<MemberId>(mi), {&t}, bt));
    }
    CompId comp = compOfMember(g, t.member);
    Value tok = rs.carrier.lookup(comp);
    assert(tok && "no carrier for view");
    Value semaVal = rs.carrierSema.lookup(comp);
    assert(semaVal && "no semaphore for carrier");
    auto buf = emitInto<nvws::SemaphoreBufferOp>(
        b, accessOp->getLoc(), owner, gpu::getStageCluster(accessOp), semaVal, TypeRange(types), tok);
    if (node->bufferStageOffset)
      buf.setStage(materializeI32Before(buf, *node->bufferStageOffset));
    for (auto [mi, v] : llvm::enumerate(buf.getBuffers()))
      if (compOfMember(g, static_cast<MemberId>(mi)) == comp)
        rs.view[static_cast<MemberId>(mi)] = v;
    base = rs.view[t.member];
  }
  Value cur = base;
  OpBuilder b(accessOp);
  for (const AliasStep &step : t.alias) {
    Operation *old = step.op;
    if (old->getName().getStringRef() == "ttg.memdesc_index" && old->getNumResults() == 1 &&
        sameViewType(old->getResult(0).getType(), cur.getType()))
      continue;
    IRMapping mapping;
    for (auto [idx, operand] : llvm::enumerate(old->getOperands()))
      mapping.map(operand, idx == step.operandIdx ? cur : operand);
    Value source = cur;
    Operation *cloned = b.clone(*old, mapping);
    if (cloned->getNumResults() == 1)
      if (auto rt = dyn_cast<gpu::MemDescType>(cloned->getResult(0).getType()))
        cloned->getResult(0).setType(withMutable(
            rt, cast<gpu::MemDescType>(source.getType()).getMutableMemory()));
    cur = cloned->getResult(0);
  }
  return cur;
}

static void renderChain(EmitCtx &ctx, GroupDag &g, Node *head, RenderState &rs,
                        DenseMap<Node *, Value> &emitted);
static Operation *renderAccess(GroupDag &g, Node *n, RenderState &rs) {
  Operation *op = n->op;
  Operation *anchor = op;
  for (const Touch &t : n->touches) {
    Value view = getView(g, rs, n, t, op, n->owner);
    if (auto ta = dyn_cast<nvidia_gpu::TMEMAllocOp>(op)) {
      OpBuilder b(op);
      auto pidsc = std::make_pair(n->owner, gpu::getStageCluster(op));
      auto vTrue = emitInto<arith::ConstantOp>(b, op->getLoc(), n->owner, pidsc.second, b.getBoolAttr(true));
      anchor = emitInto<nvidia_gpu::TMEMStoreOp>(b, op->getLoc(), n->owner, pidsc.second, Type(), view,
                                                 Value(), ta.getSrc(), vTrue);
      ta.getResult().replaceUsesWithIf(view, [&](OpOperand &use) {
        return !isa<nvws::SemaphoreCreateOp>(use.getOwner()) &&
               use.getOwner() != view.getDefiningOp() && !g.accessRowOps.contains(use.getOwner());
      });
      return anchor;
    }
    if (auto la = dyn_cast<gpu::LocalAllocOp>(op)) {
      OpBuilder b(op);
      Value src = la.getSrc();
      if (src && !isa<RankedTensorType>(src.getType())) {
        auto splat = emitInto<triton::SplatOp>(b, op->getLoc(), n->owner, gpu::getStageCluster(op),
            RankedTensorType::get(cast<gpu::MemDescType>(view.getType()).getShape(), src.getType()), src);
        src = splat.getResult();
      }
      anchor = emitInto<gpu::LocalStoreOp>(b, op->getLoc(), n->owner, gpu::getStageCluster(op), src, view);
      la.getResult().replaceUsesWithIf(view, [&](OpOperand &use) {
        return !isa<nvws::SemaphoreCreateOp>(use.getOwner()) && !g.accessRowOps.contains(use.getOwner());
      });
      return anchor;
    }
    if (auto mma = dyn_cast<nvidia_gpu::MMAv5OpInterface>(op)) {
      if (mma.getAccumulator() == t.accessValue)
        mma.setAccumulator(view);
      for (OpOperand &o : op->getOpOperands())
        if (o.get() == t.accessValue && o.get() != mma.getAccumulator())
          o.set(view);
      continue;
    }
    for (OpOperand &o : op->getOpOperands())
      if (o.get() == t.accessValue)
        o.set(view);
  }
  if (n->completionAnchor)
    anchor = n->completionAnchor;
  return anchor;
}

static void renderRegion(EmitCtx &ctx, GroupDag &g, Node *n, RenderState &rs,
                         DenseMap<Node *, Value> &emitted) {
  for (const Crossing &c : n->crossings) {
    if (n->kind == Node::For && !c.hold.materializesCarrier())
      continue; // native point-of-use: no slot (plan M2)
    unsigned idx = slotIndexFor(ctx, n->op, &g, c.comp);
    Value incoming = rs.carrier.lookup(c.comp);
    if (!incoming)
      incoming = ctx.poison; // component starts inside this region
    if (auto forOp = dyn_cast<scf::ForOp>(n->op))
      forOp.getInitsMutable()[idx].assign(incoming);
  }
  if (!n->requiredParts.empty() && gpu::hasPartition(n->op)) {
    SetVector<int> set = gpu::getPartitionIds(n->op);
    unsigned before = set.size();
    for (int p : n->requiredParts)
      set.insert(p);
    if (set.size() != before)
      gpu::setPartition(n->op, set.getArrayRef());
  }
  if (auto forOp = dyn_cast<scf::ForOp>(n->op)) {
    RenderState body = rs.nested();
    for (const Crossing &c : n->crossings) {
      if (!c.hold.materializesCarrier()) {
        body.carrier.erase(c.comp);
        continue;
      }
      unsigned idx = slotIndexFor(ctx, n->op, &g, c.comp);
      body.carrier[c.comp] = forOp.getRegionIterArg(idx);
    }
    renderChain(ctx, g, n->children[0], body, emitted);
    auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    for (const Crossing &c : n->crossings) {
      if (!c.hold.materializesCarrier()) {
        rs.carrier.erase(c.comp); // token died in the body; nothing flows out
        continue;
      }
      unsigned idx = slotIndexFor(ctx, n->op, &g, c.comp);
      yield->setOperand(idx, body.carrier.lookup(c.comp));
      rs.carrier[c.comp] = forOp.getResult(idx);
    }
    rs.clearViews();
    return;
  }
  auto ifOp = cast<scf::IfOp>(n->op);
  RenderState thenSt = rs.nested(), elseSt = rs.nested();
  renderChain(ctx, g, n->children[0], thenSt, emitted);
  if (n->children.size() > 1 && n->children[1])
    renderChain(ctx, g, n->children[1], elseSt, emitted);
  for (const Crossing &c : n->crossings) {
    unsigned idx = slotIndexFor(ctx, n->op, &g, c.comp);
    Value incoming = rs.carrier.lookup(c.comp);
    if (!incoming)
      incoming = ctx.poison;
    Value thenV = c.finals[0] ? thenSt.carrier.lookup(c.comp) : incoming;
    Value elseV = (c.finals.size() > 1 && c.finals[1]) ? elseSt.carrier.lookup(c.comp) : incoming;
    auto thenYield = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    thenYield->setOperand(idx, thenV);
    auto elseYield = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
    elseYield->setOperand(idx, elseV);
    rs.carrier[c.comp] = ifOp.getResult(idx);
  }
  rs.clearViews();
}

static void renderChain(EmitCtx &ctx, GroupDag &g, Node *head, RenderState &rs,
                        DenseMap<Node *, Value> &emitted) {
  Operation *lastReal = nullptr;
  for (Node *n = head; n; n = n->next) {
    switch (n->kind) {
    case Node::Enter:
    case Node::Exit:
      break; // markers; yield wiring is the parent's job
    case Node::Acquire: {
      CompId comp = getSema(g, n).component;
      if (Value v = emitted.lookup(n)) { // pre-rendered entry instance
        rs.carrier[comp] = v;
        rs.carrierSema[comp] = getSema(g, n).create;
        rs.clearViews();
        break;
      }
      Operation *before = nextRealOp(n->next);
      OpBuilder b(ctx.func);
      if (before) {
        b.setInsertionPoint(before);
      } else if (lastReal && !isa<triton::FuncOp>(lastReal->getBlock()->getParentOp())) {
        b.setInsertionPoint(lastReal->getBlock()->getTerminator());
      } else if (lastReal) {
        b.setInsertionPointAfter(lastReal);
      } else if (n->parent && n->parent->op) {
        Region &region = n->parent->op->getRegion(0);
        b.setInsertionPoint(region.front().getTerminator());
      }
      auto acq = emitInto<nvws::SemaphoreAcquireOp>(
          b, before ? before->getLoc() : ctx.func.getLoc(), n->owner,
          n->stageCluster, getSema(g, n).create, ctx.tokenType);
      if (n->stageOffset)
        acq.setStage(materializeI32Before(acq, *n->stageOffset));
      emitted[n] = acq.getToken();
      rs.carrier[comp] = acq.getToken();
      rs.carrierSema[comp] = getSema(g, n).create;
      rs.clearViews();
      lastReal = acq;
      break;
    }
    case Node::Release: {
      CompId comp = getSema(g, n).component;
      Value tok = rs.carrier.lookup(comp);
      assert(tok && "release without carrier");
      OpBuilder b(ctx.func);
      if (lastReal)
        b.setInsertionPointAfter(lastReal);
      else if (n->parent && n->parent->op)
        b.setInsertionPointToStart(&n->parent->op->getRegion(0).front());
      else
        b.setInsertionPointToStart(&ctx.func.getBody().front());
      auto rel = emitInto<nvws::SemaphoreReleaseOp>(
          b, lastReal ? lastReal->getLoc() : ctx.func.getLoc(), n->owner,
          n->stageCluster, getSema(g, n).create, tok, asyncOpsAttr(b.getContext(), n));
      if (n->stageOffset)
        rel.setStage(materializeI32Before(rel, *n->stageOffset));
      rel.setArriveCountAttr(b.getI32IntegerAttr(n->count));
      emitted[n] = Value();
      lastReal = rel;
      break;
    }
    case Node::Access: {
      if (Operation *anchor = renderAccess(g, n, rs))
        lastReal = anchor;
      break;
    }
    case Node::For:
    case Node::If:
      renderRegion(ctx, g, n, rs, emitted);
      lastReal = n->op;
      break;
    case Node::Func:
      break;
    }
  }
}
static bool hasPlannedBufferCopy(const GroupDag &g) {
  return llvm::any_of(g.pieceTable.members, [](const Member &m) {
    return m.allocOp->hasAttr("buffer.copy");
  });
}
static void coalesceBackings(GroupDag &g) {
  if (g.semas.empty() || g.pieceTable.members.size() < 2)
    return;
  if (!g.isTmem() && !hasPlannedBufferCopy(g))
    return;
  unsigned cover = 0;
  for (auto [i, m] : llvm::enumerate(g.pieceTable.members)) {
    const Member &c = g.pieceTable.members[cover];
    if (m.offset <= c.offset && m.offset + m.extent >= c.offset + c.extent)
      cover = i;
  }
  const Member &cm = g.pieceTable.members[cover];
  Value coverBacking = g.backing[cover];
  for (auto [i, m] : llvm::enumerate(g.pieceTable.members)) {
    if (i == cover)
      continue;
    if (m.offset < cm.offset || m.offset + m.extent > cm.offset + cm.extent)
      continue; // not contained; leave as-is
    Value backing = g.backing[i];
    Operation *alloc = backing.getDefiningOp();
    if (!g.isTmem() && (m.offset != cm.offset || backing.getType() != coverBacking.getType()))
      continue;
    OpBuilder b(coverBacking.getContext());
    b.setInsertionPointAfterValue(coverBacking);
    Value repl;
    if (m.offset == cm.offset && backing.getType() == coverBacking.getType()) {
      repl = coverBacking;
    } else {
      auto sub = nvidia_gpu::TMEMSubSliceOp::create(b, alloc->getLoc(), coverBacking,
          static_cast<int32_t>(m.offset - cm.offset), /*sizeHint*/ cast<gpu::MemDescType>(backing.getType())
              .getShape()
              .back());
      repl = sub.getResult();
      if (repl.getType() != backing.getType()) {
        auto re = gpu::MemDescReinterpretOp::create(b, alloc->getLoc(), backing.getType(), repl);
        repl = re.getResult();
      }
    }
    backing.replaceAllUsesWith(repl);
    alloc->erase();
    g.backing[i] = repl;
  }
}
static FailureOr<Value> createMixedDepthTmemView(Value ownerBacking, Value reuserBacking, int64_t offset) {
  auto ownerType = cast<gpu::MemDescType>(ownerBacking.getType());
  auto reuserType = cast<gpu::MemDescType>(reuserBacking.getType());
  ArrayRef<int64_t> ownerShape = ownerType.getShape();
  ArrayRef<int64_t> reuserShape = reuserType.getShape();
  Operation *reuserAlloc = reuserBacking.getDefiningOp();
  if (ownerShape.empty() || reuserShape.empty())
    return semaError(reuserAlloc) << "mixed-depth TMEM backing has empty shape";
  int64_t ownerBlockN = ownerShape.back();
  int64_t reuserBlockN = reuserShape.back();
  if (ownerBlockN < reuserBlockN || ownerBlockN % reuserBlockN != 0 ||
      offset < 0 || offset + reuserBlockN > ownerBlockN)
    return semaError(reuserAlloc) << "mixed-depth TMEM reuser is outside its physical owner";
  unsigned ownerWidth = ownerType.getElementTypeBitWidth();
  unsigned reuserWidth = reuserType.getElementTypeBitWidth();
  int64_t sliceN = 0;
  if (ownerWidth == reuserWidth)
    sliceN = reuserBlockN;
  else if (ownerWidth == 2 * reuserWidth)
    sliceN = reuserBlockN / 2;
  else
    return semaError(reuserAlloc) << "unsupported mixed-depth TMEM element-width reinterpretation";
  if (sliceN <= 0)
    return semaError(reuserAlloc) << "invalid mixed-depth TMEM subslice width";
  OpBuilder b(ownerBacking.getContext());
  b.setInsertionPointAfterValue(ownerBacking);
  auto sub = nvidia_gpu::TMEMSubSliceOp::create(
      b, reuserAlloc->getLoc(), ownerBacking, static_cast<int32_t>(offset), static_cast<int32_t>(sliceN));
  auto reinterpreted = gpu::MemDescReinterpretOp::create(
      b, reuserAlloc->getLoc(), reuserType, sub.getResult());
  return reinterpreted.getResult();
}
static LogicalResult
coalesceMixedDepthTmemBackings(MutableArrayRef<GroupDag> groups) {
  llvm::MapVector<int64_t, SmallVector<GroupDag *, 2>> sets;
  for (GroupDag &group : groups)
    if (group.mixedDepthPhysicalAlias && !group.semas.empty())
      sets[group.bufferId].push_back(&group);
  for (auto &[bufferId, set] : sets) {
    (void)bufferId;
    if (set.size() != 2)
      return semaError(set.front()->pieceTable.members.front().allocOp)
             << "mixed-depth TMEM reuse requires exactly two logical channels";
    bool firstOwns = canOwnMixedDepthTmem(*set[0], *set[1]);
    bool secondOwns = canOwnMixedDepthTmem(*set[1], *set[0]);
    if (firstOwns == secondOwns)
      return semaError(set.front()->pieceTable.members.front().allocOp)
             << "mixed-depth TMEM reuse has no unique physical owner by span and element width";
    GroupDag *owner = firstOwns ? set[0] : set[1];
    GroupDag *reuser = firstOwns ? set[1] : set[0];
    Value ownerBacking = owner->backing.front();
    Operation *ownerAlloc = ownerBacking.getDefiningOp();
    Value oldBacking = reuser->backing.front();
    Operation *oldAlloc = oldBacking.getDefiningOp();
    if (ownerAlloc->getBlock() != oldAlloc->getBlock() || !ownerAlloc->isBeforeInBlock(oldAlloc))
      return semaError(oldAlloc) << "mixed-depth TMEM physical owner does not dominate its reuser";
    int64_t offset = reuser->pieceTable.members.front().offset -
        owner->pieceTable.members.front().offset;
    FailureOr<Value> view = createMixedDepthTmemView(ownerBacking, oldBacking, offset);
    if (failed(view))
      return failure();
    oldBacking.replaceAllUsesWith(*view);
    reuser->backing.front() = *view;
    if (oldAlloc->use_empty())
      oldAlloc->erase();
  }
  return success();
}
static Value materializeI32Before(Operation *op, int64_t value) {
  OpBuilder b(op);
  auto cst = emitInto<arith::ConstantOp>(b, op->getLoc(), resolveOwner(op),
                                         gpu::getStageCluster(op), b.getI32IntegerAttr(value));
  return cst.getResult();
}
static LogicalResult foldCircularBackingsAndCreates(ArrayRef<GroupDag *> set) {
  if (set.size() < 2)
    return success();
  GroupDag *baseGroup = set.front();
  for (GroupDag *g : set)
    if (g->pieceTable.members.front().circularStart == 0) {
      baseGroup = g;
      break;
    }
  Value base = baseGroup->backing.front();
  for (GroupDag *g : set) {
    Value backing = g->backing.front();
    if (backing == base)
      continue;
    if (backing.getType() != base.getType())
      return semaError(backing.getDefiningOp()) << "circular logical backing type mismatch";
    backing.replaceAllUsesWith(base);
    g->backing.front() = base;
    if (Operation *op = backing.getDefiningOp())
      if (op->use_empty())
        op->erase();
  }
  nvws::SemaphoreCreateOp primaryReleased;
  nvws::SemaphoreCreateOp primaryUnreleased;
  for (GroupDag *g : set) {
    for (Sema &s : g->semas) {
      auto create = s.create.getDefiningOp<nvws::SemaphoreCreateOp>();
      if (!create)
        continue;
      bool isReleased = create.getIsReleased();
      nvws::SemaphoreCreateOp &primary = isReleased ? primaryReleased : primaryUnreleased;
      if (!primary) {
        primary = create;
        continue;
      }
      auto lhs = primary.getPendingCountAttr();
      auto rhs = create.getPendingCountAttr();
      if (!lhs || !rhs || lhs.getInt() != rhs.getInt())
        return semaError(create) << "circular folded semaphores disagree on pending_count";
      create.getResult().replaceAllUsesWith(primary.getResult());
      s.create = primary.getResult();
      if (create->use_empty())
        create.erase();
    }
  }
  return success();
}

static LogicalResult foldCircularGroups(MutableArrayRef<GroupDag> groups) {
  llvm::MapVector<int64_t, SmallVector<GroupDag *, 4>> byBufferId;
  for (GroupDag &g : groups)
    if (g.isCircular() && !g.semas.empty())
      byBufferId[g.bufferId].push_back(&g);
  for (auto &[id, set] : byBufferId) {
    (void)id;
    if (failed(foldCircularBackingsAndCreates(set)))
      return failure();
  }
  return success();
}

struct IfSplitCandidate {
  scf::IfOp ifOp;
  bool branchIsThen = true;
  nvws::SemaphoreReleaseOp releaseOp;
  nvws::SemaphoreAcquireOp acquireOp;
  unsigned tokenResultIdx = 0;
  bool releaseOnly = false;
};
static nvws::SemaphoreReleaseOp findBranchReleaseForSplit(Block *block) {
  for (Operation &op : *block) {
    if (isa<scf::YieldOp>(op))
      return nullptr;
    if (auto rel = dyn_cast<nvws::SemaphoreReleaseOp>(&op))
      return rel;
    if (isa<nvws::SemaphoreAcquireOp>(op))
      return nullptr;
    if (op.hasTrait<OpTrait::ConstantLike>() || isSupportedAliasOp(&op))
      continue;
    return nullptr;
  }
  return nullptr;
}
static nvws::SemaphoreAcquireOp findBranchTrailingAcquire(Block *block) {
  return dyn_cast_or_null<nvws::SemaphoreAcquireOp>(block->getTerminator()->getPrevNode());
}
static bool branchHasAcquireAfter(nvws::SemaphoreReleaseOp rel) {
  for (Operation *op = rel->getNextNode(); op; op = op->getNextNode()) {
    if (isa<scf::YieldOp>(op))
      return false;
    if (isa<nvws::SemaphoreAcquireOp>(op))
      return true;
  }
  return false;
}
static gpu::StageCluster inferPrecedingMmaStage(scf::IfOp ifOp) {
  for (Operation *op = ifOp->getPrevNode(); op; op = op->getPrevNode())
    if (isa<nvidia_gpu::MMAv5OpInterface>(op))
      return gpu::getStageCluster(op);
  return {};
}
static bool semaUsesTmem(Value sem) {
  auto ty = dyn_cast<nvws::SemaphoreType>(sem.getType());
  if (!ty || ty.getBaseType().empty())
    return false;
  auto md = dyn_cast<gpu::MemDescType>(ty.getBaseType()[0]);
  return md && isa<nvidia_gpu::TensorMemorySpaceAttr>(md.getMemorySpace());
}
static unsigned semaBaseTypeCount(Value sem) {
  auto ty = dyn_cast<nvws::SemaphoreType>(sem.getType());
  return ty ? ty.getBaseType().size() : 0;
}
static void assignStageIfKnown(OpBuilder &b, Operation *op, gpu::StageCluster sc) {
  if (sc)
    gpu::setStageCluster(b, op, sc);
}
static std::optional<unsigned> yieldedTokenIndex(scf::YieldOp yield, Value token) {
  for (auto [index, value] : llvm::enumerate(yield.getOperands()))
    if (value == token)
      return index;
  return std::nullopt;
}

static void workaroundLoopScheduler(EmitCtx &ctx) {
  SmallVector<IfSplitCandidate> candidates;
  ctx.func.walk([&](scf::IfOp ifOp) {
    if (ifOp.thenBlock()->empty())
      return;
    auto makeCandidate = [&](bool branchIsThen, bool releaseOnly) -> std::optional<IfSplitCandidate> {
      if (!branchIsThen && ifOp.getElseRegion().empty())
        return std::nullopt;
      Block *block = branchIsThen ? ifOp.thenBlock() : ifOp.elseBlock();
      auto rel = findBranchReleaseForSplit(block);
      if (!rel)
        return std::nullopt;
      if (releaseOnly) {
        if (!(semaUsesTmem(rel.getSemaphore()) && branchHasAcquireAfter(rel)))
          return std::nullopt;
        return IfSplitCandidate{ifOp, branchIsThen, rel, {}, 0, true};
      }
      auto acq = findBranchTrailingAcquire(block);
      if (!acq)
        return std::nullopt;
      if (semaUsesTmem(rel.getSemaphore()) && semaBaseTypeCount(rel.getSemaphore()) > 1)
        return std::nullopt;
      auto yieldOp = branchIsThen ? ifOp.thenYield() : ifOp.elseYield();
      std::optional<unsigned> pos = yieldedTokenIndex(yieldOp, acq.getToken());
      if (!pos)
        return std::nullopt;
      return IfSplitCandidate{ifOp, branchIsThen, rel, acq, *pos, false};
    };
    for (bool releaseOnly : {false, true})
      for (bool branchIsThen : {true, false})
        if (auto c = makeCandidate(branchIsThen, releaseOnly)) {
          candidates.push_back(*c);
          return;
        }
    if (auto acq = dyn_cast_or_null<nvws::SemaphoreAcquireOp>(&ifOp.thenBlock()->front())) {
      Operation *prev = ifOp->getPrevNode();
      if (prev && ifOp.getCondition().getDefiningOp() == prev)
        prev = prev->getPrevNode();
      if (auto rel = dyn_cast_or_null<nvws::SemaphoreReleaseOp>(prev)) {
        std::optional<unsigned> pos = yieldedTokenIndex(ifOp.thenYield(), acq.getToken());
        if (pos)
          candidates.push_back(IfSplitCandidate{ifOp, true, rel, acq, *pos, false});
      }
    }
  });
  for (IfSplitCandidate &c : candidates) {
    scf::IfOp ifOp = c.ifOp;
    OpBuilder b(ifOp);
    Location loc = ifOp.getLoc();
    auto exitIf = scf::IfOp::create(b, loc, TypeRange{}, ifOp.getCondition(),
                                    /*withElseRegion=*/!c.branchIsThen);
    Block *exitBlock = c.branchIsThen ? exitIf.thenBlock() : exitIf.elseBlock();
    c.releaseOp->moveBefore(exitBlock, exitBlock->begin());
    exitIf->setAttrs(ifOp->getAttrs());
    gpu::StageCluster releaseStage = gpu::getStageCluster(c.releaseOp);
    if (!releaseStage)
      releaseStage = inferPrecedingMmaStage(ifOp);
    assignStageIfKnown(b, c.releaseOp, releaseStage);
    assignStageIfKnown(b, exitIf, releaseStage);
    SetVector<int> exitIds = partitionIdsOfFwd(c.releaseOp);
    if (exitIds.empty())
      exitIds = partitionIdsOfFwd(ifOp);
    if (!exitIds.empty())
      gpu::setPartition(exitIf, exitIds.getArrayRef());
    gpu::setPartitionOutputs(exitIf, {});
    if (c.releaseOnly)
      continue;
    b.setInsertionPointAfter(ifOp);
    auto enterIf = scf::IfOp::create(b, loc, TypeRange{ctx.tokenType}, ifOp.getCondition(),
                                     /*withElseRegion=*/true);
    Block *acqBlock = c.branchIsThen ? enterIf.thenBlock() : enterIf.elseBlock();
    c.acquireOp->moveBefore(acqBlock, acqBlock->begin());
    ifOp.getResult(c.tokenResultIdx).replaceAllUsesWith(enterIf.getResult(0));
    b.setInsertionPointToEnd(enterIf.thenBlock());
    scf::YieldOp::create(b, loc, ValueRange{c.branchIsThen ? Value(c.acquireOp.getToken())
                       : ifOp.thenYield().getOperand(c.tokenResultIdx)});
    b.setInsertionPointToEnd(enterIf.elseBlock());
    scf::YieldOp::create(b, loc, ValueRange{c.branchIsThen ? ifOp.elseYield().getOperand(c.tokenResultIdx)
                       : Value(c.acquireOp.getToken())});
    b.setInsertionPoint(ifOp);
    Value poison = ub::PoisonOp::create(b, loc, ctx.tokenType).getResult();
    ifOp.thenYield().setOperand(c.tokenResultIdx, poison);
    ifOp.elseYield().setOperand(c.tokenResultIdx, poison);
    enterIf->setAttrs(ifOp->getAttrs());
    gpu::StageCluster acquireStage = gpu::getStageCluster(c.acquireOp);
    assignStageIfKnown(b, enterIf, acquireStage);
    SetVector<int> enterExitIds = partitionIdsOfFwd(c.releaseOp);
    for (int p : partitionIdsOfFwd(c.acquireOp))
      enterExitIds.insert(p);
    if (!enterExitIds.empty()) {
      gpu::setPartition(exitIf, enterExitIds.getArrayRef());
      gpu::setPartition(enterIf, enterExitIds.getArrayRef());
      gpu::setPartitionOutputs(exitIf, {});
      SmallVector<SetVector<int>, 1> enterOutputs{enterExitIds};
      gpu::setPartitionOutputs(enterIf, enterOutputs);
    }
    SetVector<int> middleIds;
    for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
      if (region->empty())
        continue;
      for (Operation &op : region->front()) {
        if (isa<scf::YieldOp>(op))
          continue;
        for (int p : partitionIdsOfFwd(&op))
          middleIds.insert(p);
      }
    }
    auto authored = gpu::getPartitionOutputs(ifOp);
    for (auto [i, o] : llvm::enumerate(authored)) {
      if (i == c.tokenResultIdx)
        continue; // dead after the reroute; dropped by the sweep
      for (int p : o)
        middleIds.insert(p);
    }
    if (!middleIds.empty())
      gpu::setPartition(ifOp, middleIds.getArrayRef());
  }
}

static LogicalResult verifyPartitionOutputs(triton::FuncOp func) {
  constexpr llvm::StringLiteral kOutputs = "ttg.partition.outputs";
  auto producerIds = [&](Value v) -> SetVector<int> {
    Operation *def = v.getDefiningOp();
    if (!def || isa<ub::PoisonOp>(def) || def->hasTrait<OpTrait::ConstantLike>())
      return {};
    if (!gpu::hasPartition(def))
      return {};
    return gpu::getPartitionIds(def);
  };
  LogicalResult result = success();
  func.walk([&](Operation *op) {
    if (!isa<scf::ForOp, scf::IfOp>(op) || !op->hasAttr(kOutputs) || failed(result))
      return;
    auto outputs = gpu::getPartitionOutputs(op);
    if (outputs.size() != op->getNumResults()) {
      result = semaError(op) << "partition-outputs verifier: attribute has "
               << outputs.size() << " entries for " << op->getNumResults() << " results";
      return;
    }
    SmallVector<Operation *, 2> terms;
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      terms.push_back(forOp.getBody()->getTerminator());
    } else {
      auto ifOp = cast<scf::IfOp>(op);
      terms.push_back(ifOp.thenYield());
      if (!ifOp.getElseRegion().empty())
        terms.push_back(ifOp.elseYield());
    }
    for (auto [i, outSet] : llvm::enumerate(outputs))
      for (Operation *term : terms) {
        SetVector<int> prod = producerIds(term->getOperand(i));
        if (prod.empty())
          continue; // indeterminate producer: out of scope, never guessed
        if (llvm::none_of(prod, [&](int p) { return outSet.contains(p); })) {
          std::string have;
          llvm::raw_string_ostream os(have);
          for (int p : prod)
            os << p << " ";
          result = semaError(op) << "partition-outputs verifier: result "
                   << i << " is produced by partition(s) " << os.str()
                   << "but ttg.partition.outputs names none of them";
          return;
        }
      }
  });
  return result;
}

static LogicalResult verifyTokenLocality(triton::FuncOp func) {
  auto idsOf = [](Operation *op) -> std::optional<SmallVector<int, 2>> {
    if (!gpu::hasPartition(op))
      return std::nullopt;
    auto set = gpu::getPartitionIds(op);
    SmallVector<int, 2> v(set.begin(), set.end());
    llvm::sort(v);
    return v;
  };
  std::function<LogicalResult(Operation *, Value, DenseSet<Value> &)> trace =
      [&](Operation *consumer, Value tok, DenseSet<Value> &seen) -> LogicalResult {
    if (!seen.insert(tok).second)
      return success();
    if (auto ba = dyn_cast<BlockArgument>(tok)) {
      if (auto forOp = dyn_cast<scf::ForOp>(ba.getOwner()->getParentOp())) {
        unsigned idx = ba.getArgNumber() - 1; // skip induction var
        if (failed(trace(consumer, forOp.getInits()[idx], seen)))
          return failure();
        auto *yield = forOp.getBody()->getTerminator();
        return trace(consumer, yield->getOperand(idx), seen);
      }
      return success(); // other block args: out of scope
    }
    Operation *def = tok.getDefiningOp();
    if (!def || isa<ub::PoisonOp>(def))
      return success();
    if (auto acq = dyn_cast<nvws::SemaphoreAcquireOp>(def)) {
      auto ap = idsOf(acq), cp = idsOf(consumer);
      if (!ap)
        return success(); // root entry seed exemption
      if (cp && *ap != *cp) {
        InFlightDiagnostic diag = semaError(consumer) << "token-locality violation: token acquired "
                                     "in partition set differs from consumer's";
        diag.attachNote(acq.getLoc()) << "acquired here";
        return failure();
      }
      return success();
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
      unsigned idx = cast<OpResult>(tok).getResultNumber();
      if (failed(trace(consumer, ifOp.thenYield().getOperand(idx), seen)))
        return failure();
      if (!ifOp.getElseRegion().empty())
        return trace(consumer, ifOp.elseYield().getOperand(idx), seen);
      return success();
    }
    if (auto forOp = dyn_cast<scf::ForOp>(def)) {
      unsigned idx = cast<OpResult>(tok).getResultNumber();
      if (failed(trace(consumer, forOp.getInits()[idx], seen)))
        return failure();
      auto *yield = forOp.getBody()->getTerminator();
      return trace(consumer, yield->getOperand(idx), seen);
    }
    return success();
  };
  LogicalResult result = success();
  func.walk([&](Operation *op) {
    Value tok;
    if (auto rel = dyn_cast<nvws::SemaphoreReleaseOp>(op))
      tok = rel.getToken();
    else if (auto buf = dyn_cast<nvws::SemaphoreBufferOp>(op))
      tok = buf.getToken();
    else
      return;
    DenseSet<Value> seen;
    if (failed(trace(op, tok, seen)))
      result = failure();
    if (auto buf = dyn_cast<nvws::SemaphoreBufferOp>(op)) {
      auto bp = idsOf(op);
      if (bp)
        for (Value view : buf->getResults())
          for (Operation *user : view.getUsers()) {
            auto up = idsOf(user);
            if (up && *up != *bp) {
              InFlightDiagnostic diag = semaError(user)
                  << "view-locality violation: view consumed outside its partition";
              diag.attachNote(op->getLoc()) << "view materialized here";
              result = failure();
            }
          }
    }
  });
  return result;
}
static LogicalResult verifyNoUseAfterRelease(triton::FuncOp funcOp) {
  auto checkToken = [](Value tok) -> LogicalResult {
    llvm::SmallDenseMap<Block *, SmallVector<Operation *, 4>> byBlock;
    for (Operation *u : tok.getUsers())
      byBlock[u->getBlock()].push_back(u);
    for (auto &[block, users] : byBlock) {
      llvm::sort(users, [](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
      });
      bool sawRelease = false;
      for (Operation *u : users) {
        if (isa<nvws::SemaphoreReleaseOp>(u))
          sawRelease = true;
        else if (sawRelease && isa<nvws::SemaphoreBufferOp>(u))
          return semaError(u) << "token has a buffer view after its release "
                    "(use-after-release; spec fable/semas-report3.md Addendum B.3(b))";
      }
    }
    return success();
  };
  auto res = funcOp.walk([&](Operation *op) -> WalkResult {
    if (auto acq = dyn_cast<nvws::SemaphoreAcquireOp>(op))
      if (failed(checkToken(acq.getToken())))
        return WalkResult::interrupt();
    if (auto forOp = dyn_cast<scf::ForOp>(op))
      for (BlockArgument arg : forOp.getRegionIterArgs())
        if (isa<gpu::AsyncTokenType>(arg.getType()) && failed(checkToken(arg)))
          return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(res.wasInterrupted());
}
static nvws::SemaphoreAcquireOp resolveAcquireThroughIfs(Value v) {
  for (int fuel = 0; fuel < 8; ++fuel) {
    if (auto acq = v.getDefiningOp<nvws::SemaphoreAcquireOp>())
      return acq;
    auto ifOp = v.getDefiningOp<scf::IfOp>();
    if (!ifOp)
      return nullptr;
    unsigned idx = cast<OpResult>(v).getResultNumber();
    Value t = ifOp.thenYield()->getOperand(idx);
    if (auto acq = t.getDefiningOp<nvws::SemaphoreAcquireOp>())
      return acq;
    if (ifOp.elseBlock()) {
      Value e = ifOp.elseYield()->getOperand(idx);
      if (auto acq = e.getDefiningOp<nvws::SemaphoreAcquireOp>())
        return acq;
    }
    v = t;
  }
  return nullptr;
}
static LogicalResult verifySingleCarrierPerGroup(triton::FuncOp funcOp) {
  auto res = funcOp.walk([&](scf::ForOp forOp) -> WalkResult {
    llvm::SmallDenseMap<Value, unsigned> slotsPerBacking;
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    for (BlockArgument arg : forOp.getRegionIterArgs()) {
      if (!isa<gpu::AsyncTokenType>(arg.getType()))
        continue;
      unsigned idx = arg.getArgNumber() - 1; // skip induction variable
      nvws::SemaphoreAcquireOp acq = resolveAcquireThroughIfs(yieldOp.getOperand(idx));
      if (!acq)
        continue;
      auto create = acq.getSemaphore().getDefiningOp<nvws::SemaphoreCreateOp>();
      if (!create || create.getBuffers().empty())
        continue;
      Value backing = create.getBuffers().front();
      if (auto alloc = backing.getDefiningOp<gpu::LocalAllocOp>())
        if (alloc->hasAttr(kBufferCircularAttrName))
          continue;
      if (++slotsPerBacking[backing] > 1) {
        semaError(forOp) << "two carrier token slots for one semaphore group in a single "
               "loop (spec fable/semas-report3.md Addendum B.3(a)); " "AssignStagePhase cannot thread this";
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return failure(res.wasInterrupted());
}

LogicalResult emitIR(triton::FuncOp funcOp, MutableArrayRef<GroupDag> groups) {
  EmitCtx ctx;
  ctx.func = funcOp;
  ctx.tokenType = gpu::AsyncTokenType::get(funcOp.getContext());
  {
    OpBuilder b(&funcOp.getBody().front(), funcOp.getBody().front().begin());
    ctx.poison = ub::PoisonOp::create(b, funcOp.getLoc(), ctx.tokenType).getResult();
  }
  for (GroupDag &g : groups)
    if (!g.semas.empty()) {
      nukeGroupTokens(ctx, g);
      forEachNode(g, [&](Node *n) {
        if (n->kind == Node::Access && n->op)
          g.accessRowOps.insert(n->op);
      });
    }
  while (eraseDeadTokenSlots(ctx, groups)) {
  }
  DenseMap<Node *, Value> emitted;
  for (GroupDag &g : groups)
    emitBackingsAndCreates(ctx, g);
  for (GroupDag &g : groups)
    if (!g.semas.empty())
      emitEntryAcquires(ctx, g, emitted);
  rewriteSignatures(ctx, groups);
  for (GroupDag &g : groups) {
    if (g.semas.empty())
      continue;
    RenderState rs;
    for (Node *n = g.root->children[0]; n; n = n->next)
      if (n->kind == Node::Acquire && emitted.count(n)) {
        CompId comp = getSema(g, n).component;
        rs.carrier[comp] = emitted.lookup(n);
        rs.carrierSema[comp] = getSema(g, n).create;
      }
    renderChain(ctx, g, g.root->children[0], rs, emitted);
  }
  if (failed(foldCircularGroups(groups)))
    return failure();
  for (GroupDag &g : groups)
    coalesceBackings(g);
  if (failed(coalesceMixedDepthTmemBackings(groups)))
    return failure();
  {
    workaroundLoopScheduler(ctx);
    while (eraseDeadTokenSlots(ctx, groups)) {
    }
  }
  {
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> aliasOps;
      ctx.func.walk([&](Operation *op) {
        if (isSupportedAliasOp(op))
          aliasOps.push_back(op);
      });
      for (Operation *op : llvm::reverse(aliasOps))
        if (llvm::all_of(op->getResults(), [](Value v) { return v.use_empty(); })) {
          op->erase();
          changed = true;
        }
    }
  }
  for (GroupDag &g : groups) {
    if (g.semas.empty())
      continue;
    for (const Member &m : g.pieceTable.members)
      if (m.allocOp && m.allocOp->getBlock() && m.allocOp->use_empty())
        m.allocOp->erase();
  }
  if (ctx.poison.use_empty())
    ctx.poison.getDefiningOp()->erase();
  if (failed(verifyPartitionOutputs(funcOp)))
    return failure();
  if (failed(verifyTokenLocality(funcOp)))
    return failure();
  if (failed(verifyNoUseAfterRelease(funcOp)))
    return failure();
  if (failed(verifySingleCarrierPerGroup(funcOp)))
    return failure();
  return success();
}
} // namespace mlir::triton::nvws_semas
