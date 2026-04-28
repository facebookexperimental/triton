// SYNC analysis and scheduling; see sema-docs/insert-semas/sync-dag.md.
#include "InsertSemas.h"
#include <limits>
#include <numeric>

namespace mlir::triton::nvws_semas {

struct HolderRec {
  Owner owner;
  Node *lastRow = nullptr;
  SmallVector<AsyncOp, 1> lastPayloads;
  SmallVector<int64_t, 2> syncedBehind;
};

struct PieceGame {
  bool live = false;
  Owner versionProducer; // current version's producer (root = nullopt)
  SmallVector<HolderRec, 2> holders; // producer and/or readers, join order
};
using ChainState = std::map<PieceId, PieceGame>;

struct EdgeRec {
  Node *src = nullptr;
  Node *dst = nullptr;
  Owner srcOwner, dstOwner;
  SmallVector<AsyncOp, 1> payloads;
  SmallVector<PieceId, 2> pieces;
};
static HolderRec *findHolder(PieceGame &gm, const Owner &who) {
  for (HolderRec &h : gm.holders)
    if (sameOwner(h.owner, who))
      return &h;
  return nullptr;
}
static void unionPayloads(SmallVector<AsyncOp, 1> &into, const SmallVector<AsyncOp, 1> &from) {
  for (AsyncOp p : from)
    if (!llvm::is_contained(into, p))
      into.push_back(p);
  llvm::sort(into, [](AsyncOp a, AsyncOp b) {
    return static_cast<int>(a) < static_cast<int>(b);
  });
}
// Advance one piece's owner/reader game and emit only cross-owner dependencies.
static void applyTouch(ChainState &st, PieceId p, const Owner &who, Effect effect, Node *row,
                       const SmallVector<AsyncOp, 1> &rowPayloads,
                       SmallVector<EdgeRec> &edges, bool wsAdopt, bool force = false,
                       Node *waveSrc = nullptr, Owner waveOwner = Owner(),
                       const SmallVector<AsyncOp, 1> &wavePay = {}) {
  PieceGame &gm = st[p];
  if (!gm.live) { // first toucher in an unseeded (function-level) game
    gm.live = true;
    gm.versionProducer = who;
    gm.holders.assign(1, HolderRec{who, row, rowPayloads, {}});
    return;
  }
  if (effect == Effect::W) {
    bool edged = false;
    for (HolderRec &h : gm.holders) {
      if (sameOwner(h.owner, who))
        continue;
      if (wsAdopt && !h.owner.has_value())
        continue; // adoption: no edge from a root holder
      if (!force && llvm::is_contained(h.syncedBehind, ownerKey(who)))
        continue; // transitively synchronized — edge redundant
      edges.push_back(EdgeRec{h.lastRow, row, h.owner, who, h.lastPayloads, {p}});
      edged = true;
    }
    if (force && !edged && waveSrc && !sameOwner(waveOwner, who))
      edges.push_back(EdgeRec{waveSrc, row, waveOwner, who, wavePay, {p}});
    gm.holders.assign(1, HolderRec{who, row, rowPayloads, {}});
    gm.versionProducer = who;
    return;
  }
  if (HolderRec *h = findHolder(gm, who)) { // reread (producer or reader)
    if (force) { // wave closed: the reread must re-acquire
      HolderRec *primary = findHolder(gm, gm.versionProducer);
      if (primary && !sameOwner(primary->owner, who)) {
        edges.push_back(EdgeRec{primary->lastRow, row, primary->owner, who, primary->lastPayloads, {p}});
        primary->syncedBehind.push_back(ownerKey(who));
      } else if (waveSrc && !sameOwner(waveOwner, who)) {
        edges.push_back(EdgeRec{waveSrc, row, waveOwner, who, wavePay, {p}});
      }
    }
    h->lastRow = row;
    h->lastPayloads = rowPayloads;
    h->syncedBehind.clear(); // lastRow moved
    return;
  }
  HolderRec *primary = findHolder(gm, gm.versionProducer);
  if (!primary)
    primary = &gm.holders.front();
  if (!(wsAdopt && !primary->owner.has_value())) {
    edges.push_back(EdgeRec{primary->lastRow, row, primary->owner, who, primary->lastPayloads, {p}});
    primary->syncedBehind.push_back(ownerKey(who));
  }
  gm.holders.push_back(HolderRec{who, row, rowPayloads, {}});
}
static bool rowTouchesPiece(GroupDag &g, Node *n, PieceId piece) {
  if (n->kind == Node::Access)
    return touchesPiece(g, n, piece);
  if (n->isRegion())
    return n->pieceInfo.count(piece) > 0;
  return false;
}
static bool pieceTouchedAfter(GroupDag &g, Node *regionRow, PieceId piece) {
  for (Node *r = regionRow; r && r->kind != Node::Func;) {
    for (Node *m = r->next; m; m = m->next)
      if (rowTouchesPiece(g, m, piece))
        return true;
    r = r->parent;
  }
  return false;
}
static std::map<PieceId, SmallVector<AsyncOp, 1>>
walkChain(GroupDag &g, Node *head, ChainState &st, SmallVector<EdgeRec> &edges, bool underFor) {
  std::map<PieceId, Owner> carried;
  if (head->kind == Node::Enter)
    for (auto &[p, pi] : sortedPieceInfo(head))
      carried[p] = pi.owner;
  struct WaveSt {
    Owner owner;
    bool valid = false;
    Node *lastRow = nullptr;
    SmallVector<AsyncOp, 1> pay;
  };
  std::map<CompId, WaveSt> wave;
  auto compOf = [&](PieceId p) { return g.pieceTable.pieceComp[p]; };
  if (head->kind == Node::Enter)
    for (auto &[p, pi] : sortedPieceInfo(head)) {
      WaveSt &w = wave[compOf(p)];
      if (!w.lastRow) {
        w.owner = pi.owner;
        w.valid = true;
        w.lastRow = head;
        w.pay.assign(1, AsyncOp::NONE);
      } else if (!sameOwner(w.owner, pi.owner))
        w.valid = false;
    }
  for (Node *n = head; n; n = n->next) {
    switch (n->kind) {
    case Node::Enter:
      break; // seeding happened at chain entry
    case Node::Access: {
      std::map<PieceId, Effect> eff;
      forEachTouchedPiece(g, n, [&](PieceId p, Effect e) { mergeEffect(eff, p, e); });
      for (auto &[p, e] : eff) {
        SmallVector<AsyncOp, 1> pay;
        pay.push_back(asyncPayloadOf(n->op));
        WaveSt &w = wave[compOf(p)];
        bool force = w.valid && n->owner.has_value() && w.owner.has_value() && !sameOwner(w.owner, n->owner);
        applyTouch(st, p, n->owner, e, n, pay, edges, /*wsAdopt=*/false, force, w.lastRow, w.owner, w.pay);
        if (n->owner.has_value()) {
          w.owner = n->owner;
          w.valid = true;
          w.lastRow = n;
          w.pay.assign(1, asyncPayloadOf(n->op));
        }
      }
      break;
    }
    case Node::For:
    case Node::If: {
      bool wsAdopt = n->kind == Node::For && gpu::hasWarpSpecializeTag(n->op);
      auto infos = sortedPieceInfo(n);
      std::map<PieceId, std::pair<Owner, SmallVector<AsyncOp, 1>>> preProd;
      for (auto &[p, pi] : infos) {
        auto it = st.find(p);
        if (it == st.end() || !it->second.live)
          continue;
        PieceGame &gm = it->second;
        HolderRec *ph = findHolder(gm, gm.versionProducer);
        SmallVector<AsyncOp, 1> pay;
        if (ph)
          pay = ph->lastPayloads;
        else
          pay.push_back(AsyncOp::NONE);
        preProd.emplace(p, std::make_pair(gm.versionProducer, pay));
      }
      for (auto &[p, pi] : infos) {
        SmallVector<AsyncOp, 1> none;
        none.push_back(AsyncOp::NONE); // placeholder; replaced below
        applyTouch(st, p, pi.owner, pi.effect, n, none, edges, wsAdopt);
      }
      std::map<PieceId, SmallVector<AsyncOp, 1>> unionRet;
      for (Node *childHead : n->children) {
        ChainState child;
        for (auto &[p, pi] : sortedPieceInfo(childHead)) {
          PieceGame gm;
          gm.live = true;
          Owner childCarried = pi.owner;
          auto pre = preProd.find(p);
          gm.versionProducer = pre != preProd.end() ? pre->second.first : childCarried;
          SmallVector<AsyncOp, 1> seedPay;
          if (pre != preProd.end() && sameOwner(childCarried, pre->second.first))
            seedPay = pre->second.second; // payload-seed IMPORT
          else
            seedPay.push_back(AsyncOp::NONE); // transitivity witness
          gm.holders.assign(1, HolderRec{childCarried, childHead, seedPay, {}});
          child.emplace(p, std::move(gm));
        }
        auto ret = walkChain(g, childHead, child, edges, underFor || n->kind == Node::For);
        for (auto &[p, pay] : ret)
          unionPayloads(unionRet[p], pay);
      }
      for (auto &[p, pi] : infos) {
        PieceGame &gm = st[p];
        if (HolderRec *h = findHolder(gm, pi.owner)) {
          auto it = unionRet.find(p);
          if (it != unionRet.end())
            h->lastPayloads = it->second;
        }
      }
      {
        std::map<CompId, std::pair<Owner, bool>> ro; // owner, uniform
        for (auto &[p, pi] : infos) {
          auto [it, fresh] = ro.try_emplace(compOf(p), std::make_pair(pi.owner, true));
          if (!fresh && !sameOwner(it->second.first, pi.owner))
            it->second.second = false;
        }
        for (auto &[c, ou] : ro) {
          WaveSt &w = wave[c];
          if (ou.second && ou.first.has_value()) {
            w.owner = ou.first;
            w.valid = true;
            w.lastRow = n;
            w.pay.assign(1, AsyncOp::NONE);
          } else {
            w.valid = false;
          }
        }
      }
      break;
    }
    case Node::Exit: {
      for (auto &[p, pi] : sortedPieceInfo(n)) {
        auto it = st.find(p);
        if (it == st.end())
          continue;
        PieceGame &gm = it->second;
        bool needed = underFor || pieceTouchedAfter(g, n->parent, p);
        if (needed)
          for (HolderRec &h : gm.holders) {
            if (sameOwner(h.owner, pi.owner))
              continue;
            if (llvm::is_contained(h.syncedBehind, ownerKey(pi.owner)))
              continue; // carried owner already synchronized — no close
            edges.push_back(EdgeRec{h.lastRow, n, h.owner, pi.owner, h.lastPayloads, {p}});
          }
        HolderRec keep;
        if (HolderRec *ch = findHolder(gm, pi.owner))
          keep = *ch;
        else
          keep = HolderRec{pi.owner, n, {AsyncOp::NONE}, {}};
        gm.holders.assign(1, keep);
        gm.versionProducer = pi.owner;
      }
      break;
    }
    case Node::Acquire:
    case Node::Release:
    case Node::Func:
      break; // not present during the walk
    }
  }
  std::map<PieceId, SmallVector<AsyncOp, 1>> result;
  for (auto &[p, who] : carried) {
    auto it = st.find(p);
    SmallVector<AsyncOp, 1> pay;
    if (it != st.end())
      if (HolderRec *h = findHolder(it->second, who))
        pay = h->lastPayloads;
    if (pay.empty())
      pay.push_back(AsyncOp::NONE);
    result.emplace(p, pay);
  }
  return result;
}
static void spliceBefore(Node *node, Node *before) {
  node->parent = before->parent;
  node->prev = before->prev;
  node->next = before;
  if (before->prev)
    before->prev->next = node;
  else if (node->parent) // chain head: repoint the parent's children slot
    for (Node *&slot : node->parent->children)
      if (slot == before)
        slot = node;
  before->prev = node;
}
static void spliceAfter(Node *node, Node *after) {
  node->parent = after->parent;
  node->next = after->next;
  node->prev = after;
  if (after->next)
    after->next->prev = node;
  after->next = node;
}
static Node *newProtocolNode(GroupDag &g, Node::Kind kind, Node *parent,
                             Owner owner, SemaId sema, unsigned count) {
  Node *node = g.newNode(kind, nullptr, parent);
  node->owner = owner;
  node->sema = sema;
  node->count = count;
  return node;
}

namespace {
using SyncVec = std::map<int64_t, unsigned>; // partitionKey -> row index

struct ChainIndex {
  DenseMap<Node *, unsigned> idx; // row -> position within its chain
  DenseMap<Node *, Node *> chainOf; // row -> chain head
};
static void indexChains(Node *head, ChainIndex &ci) {
  unsigned i = 0;
  for (Node *n = head; n; n = n->next) {
    ci.idx[n] = i++;
    ci.chainOf[n] = head;
    if (n->isRegion())
      for (Node *child : n->children)
        indexChains(child, ci);
  }
}
static bool covers(const SyncVec &v, int64_t key, unsigned srcIdx) {
  auto it = v.find(key);
  return it != v.end() && it->second >= srcIdx;
}
static LogicalResult sweepChain(GroupDag &g, Node *head, ChainIndex &ci,
                                SmallVector<EdgeRec> &edges, std::vector<bool> &drop, bool reduce,
                                ArrayRef<unsigned> checkIdxs) {
  DenseMap<Node *, SmallVector<unsigned, 2>> atDst;
  for (auto [i, e] : llvm::enumerate(edges))
    if (ci.chainOf.lookup(e.src) == head && ci.chainOf.lookup(e.dst) == head)
      atDst[e.dst].push_back(i);
  for (auto &[d, v] : atDst)
    llvm::stable_sort(v, [&](unsigned a, unsigned b) {
      return ci.idx.lookup(edges[a].src) > ci.idx.lookup(edges[b].src);
    });
  std::map<int64_t, SyncVec> behind;
  DenseMap<Node *, SyncVec> snapAtRow; // source snapshots
  std::map<CompId, int64_t> waveOf;
  if (head->kind == Node::Enter)
    for (auto &[pc, pi] : sortedPieceInfo(head))
      if (pi.owner)
        waveOf[g.pieceTable.pieceComp[pc]] = ownerKey(pi.owner);
  auto compOfEdge = [&](const EdgeRec &e) {
    return g.pieceTable.pieceComp[e.pieces.front()];
  };
  auto ownerOfRow = [&](Node *n) -> Owner {
    return n->kind == Node::Exit ? Owner() : n->owner;
  };
  for (Node *n = head; n; n = n->next) {
    auto it = atDst.find(n);
    if (it != atDst.end())
      for (unsigned ei : it->second) {
        EdgeRec &e = edges[ei];
        if (!e.srcOwner || !e.dstOwner || e.src->kind != Node::Access)
          continue; // region/root endpoints: never reduced
        int64_t sk = ownerKey(e.srcOwner), dk = ownerKey(e.dstOwner);
        unsigned srcIdx = ci.idx.lookup(e.src);
        bool implied = covers(behind[dk], sk, srcIdx);
        bool waveOpen = waveOf.count(compOfEdge(e)) && waveOf[compOfEdge(e)] == dk;
        if (reduce && !drop[ei] && implied && waveOpen && e.dst->kind == Node::Access) {
          drop[ei] = true;
          continue;
        }
        if (drop[ei])
          continue;
        waveOf[compOfEdge(e)] = dk; // kept acquire opens Q's wave
        SyncVec &dv = behind[dk];
        auto snap = snapAtRow.find(e.src);
        if (snap != snapAtRow.end())
          for (auto &[k, v] : snap->second)
            if (!covers(dv, k, v))
              dv[k] = std::max(dv[k], v);
        dv[sk] = std::max(dv[sk], srcIdx);
      }
    if (Owner o = ownerOfRow(n)) {
      behind[ownerKey(o)][ownerKey(o)] = ci.idx.lookup(n);
      snapAtRow[n] = behind[ownerKey(o)];
    }
  }
  if (!reduce)
    for (unsigned ei : checkIdxs) {
      EdgeRec &e = edges[ei];
      if (ci.chainOf.lookup(e.src) != head)
        continue;
      if (e.dst->kind == Node::Exit)
        continue; // EXIT closes are re-verified by sweepTraversalClosure
      if (!covers(behind[ownerKey(e.dstOwner)], ownerKey(e.srcOwner), ci.idx.lookup(e.src)))
        return semaError(e.dst->op ? e.dst->op : g.root->op)
               << "transitive-reduction closure violation: dropped edge is "
                  "not implied by the final edge set";
    }
  return success();
}
static LogicalResult sweepTraversalClosure(GroupDag &g, Node *head, ChainIndex &ci,
                                   SmallVector<EdgeRec> &edges, std::vector<bool> &drop, bool reduce,
                                   ArrayRef<unsigned> checkIdxs) {
  Owner firstWaveOwner;
  for (Node *n = head; n && !firstWaveOwner; n = n->next)
    if (n->kind == Node::Access && n->owner)
      firstWaveOwner = n->owner;
  if (!firstWaveOwner)
    return success();
  constexpr unsigned kPass2 = 1u << 20;
  DenseMap<Node *, SmallVector<unsigned, 2>> atDst;
  SmallVector<unsigned, 4> closes;
  for (auto [i, e] : llvm::enumerate(edges)) {
    if (drop[i] || ci.chainOf.lookup(e.src) != head || ci.chainOf.lookup(e.dst) != head)
      continue;
    if (e.dst->kind == Node::Exit && e.src->kind == Node::Access && e.srcOwner && e.dstOwner)
      closes.push_back(i);
    else
      atDst[e.dst].push_back(i);
  }
  if (closes.empty())
    return success();
  for (auto &[d, v] : atDst)
    llvm::stable_sort(v, [&](unsigned a, unsigned b) {
      return ci.idx.lookup(edges[a].src) > ci.idx.lookup(edges[b].src);
    });
  auto firstTouch = [&](const Owner &q, PieceId pc) -> Node * {
    for (Node *n = head; n; n = n->next)
      if (n->kind == Node::Access && n->owner && sameOwner(n->owner, q))
        for (const Touch &t : n->touches)
          for (PieceId fp : g.pieceTable.footprint[t.member])
            if (fp == pc)
              return n;
    return nullptr;
  };
  std::map<int64_t, SyncVec> behind;
  DenseMap<Node *, SyncVec> snap1, snap2;
  std::map<int64_t, unsigned> waveOpenAt; // ownerKey -> pass-2 open row
  auto applyKept = [&](EdgeRec &e, unsigned srcIdx, DenseMap<Node *, SyncVec> &snaps) {
    SyncVec &dv = behind[ownerKey(e.dstOwner)];
    auto sn = snaps.find(e.src);
    if (sn != snaps.end())
      for (auto &[k, v] : sn->second)
        dv[k] = std::max(dv[k], v);
    int64_t sk = ownerKey(e.srcOwner);
    dv[sk] = std::max(dv[sk], srcIdx);
  };
  for (Node *n = head; n; n = n->next) {
    auto it = atDst.find(n);
    if (it != atDst.end())
      for (unsigned ei : it->second)
        applyKept(edges[ei], ci.idx.lookup(edges[ei].src), snap1);
    if (n->owner && n->kind == Node::Access) {
      behind[ownerKey(n->owner)][ownerKey(n->owner)] = ci.idx.lookup(n);
      snap1[n] = behind[ownerKey(n->owner)];
    }
  }
  DenseMap<Node *, SmallVector<unsigned, 2>> closeAt;
  for (unsigned ei : closes) {
    EdgeRec &e = edges[ei];
    Node *latest = nullptr;
    for (PieceId pc : e.pieces)
      if (Node *ft = firstTouch(e.dstOwner, pc))
        if (!latest || ci.idx.lookup(ft) > ci.idx.lookup(latest))
          latest = ft;
    if (latest)
      closeAt[latest].push_back(ei);
  }
  LogicalResult result = success();
  for (Node *n = head; n; n = n->next) {
    auto it = atDst.find(n);
    if (it != atDst.end())
      for (unsigned ei : it->second) {
        if (drop[ei])
          continue;
        applyKept(edges[ei], kPass2 + ci.idx.lookup(edges[ei].src), snap2);
        waveOpenAt.try_emplace(ownerKey(edges[ei].dstOwner), kPass2 + ci.idx.lookup(n));
      }
    auto ct = closeAt.find(n);
    if (ct != closeAt.end())
      for (unsigned ei : ct->second) {
        EdgeRec &e = edges[ei];
        int64_t dk = ownerKey(e.dstOwner);
        bool covered = covers(behind[dk], ownerKey(e.srcOwner), ci.idx.lookup(e.src));
        bool open = waveOpenAt.count(dk);
        bool isCarrierClose = sameOwner(e.dstOwner, firstWaveOwner);
        if (reduce) {
          if (!drop[ei] && covered && open && !isCarrierClose)
            drop[ei] = true;
          if (!drop[ei]) // kept close: provides its ordering at dst
            applyKept(e, ci.idx.lookup(e.src), snap1);
        } else if (!drop[ei]) {
          applyKept(e, ci.idx.lookup(e.src), snap1);
        } else if (llvm::is_contained(checkIdxs, ei) && !(covered && open)) {
          result = semaError(e.src->op ? e.src->op : g.root->op)
                   << "traversal-closure violation: dropped close not implied";
        }
      }
    if (n->owner && n->kind == Node::Access) {
      behind[ownerKey(n->owner)][ownerKey(n->owner)] = kPass2 + ci.idx.lookup(n);
      snap2[n] = behind[ownerKey(n->owner)];
    }
  }
  return result;
}

// Remove only edges re-proved by kept waits, program order, and loop closure.
static LogicalResult reduceEdges(GroupDag &g, SmallVector<EdgeRec> &edges) {
  if (g.root->children.empty() || edges.empty())
    return success();
  ChainIndex ci;
  indexChains(g.root->children[0], ci);
  std::vector<bool> drop(edges.size(), false);
  SmallVector<Node *, 8> heads;
  DenseSet<Node *> seen;
  for (auto &[row, h] : ci.chainOf)
    if (seen.insert(h).second)
      heads.push_back(h);
  llvm::sort(heads, [&](Node *a, Node *b) {
    return ci.idx.lookup(a) < ci.idx.lookup(b);
  });
  for (Node *h : heads)
    if (failed(sweepChain(g, h, ci, edges, drop, /*reduce=*/true, {})))
      return failure();
  for (Node *h : heads)
    if (h->parent && h->parent->kind == Node::For)
      if (failed(sweepTraversalClosure(g, h, ci, edges, drop, /*reduce=*/true, {})))
        return failure();
  SmallVector<unsigned, 8> dropped;
  for (auto [i, d] : llvm::enumerate(drop))
    if (d)
      dropped.push_back(i);
  if (dropped.empty())
    return success();
  for (Node *h : heads)
    if (failed(sweepChain(g, h, ci, edges, drop, /*reduce=*/false, dropped)))
      return failure();
  for (Node *h : heads)
    if (h->parent && h->parent->kind == Node::For)
      if (failed(sweepTraversalClosure(g, h, ci, edges, drop, /*reduce=*/false, dropped)))
        return failure();
  SmallVector<EdgeRec> kept;
  for (auto [i, e] : llvm::enumerate(edges))
    if (!drop[i])
      kept.push_back(e);
  edges = std::move(kept);
  return success();
}
} // namespace
static void absorbEdge(EdgeRec &dst, const EdgeRec &src) {
  unionPayloads(dst.payloads, src.payloads);
  for (PieceId piece : src.pieces)
    if (!llvm::is_contained(dst.pieces, piece))
      dst.pieces.push_back(piece);
  llvm::sort(dst.pieces);
}
static bool followsInChain(Node *node, Node *other) {
  for (Node *next = other->next; next; next = next->next)
    if (next == node)
      return true;
  return false;
}
template <typename KeyFn>
static SmallVector<EdgeRec> mergeEdges(ArrayRef<EdgeRec> edges, KeyFn key, bool keepLatestSource) {
  using Key = decltype(key(std::declval<const EdgeRec &>()));
  DenseMap<Key, unsigned> index;
  SmallVector<EdgeRec> merged;
  for (const EdgeRec &edge : edges) {
    auto [it, inserted] = index.try_emplace(key(edge), merged.size());
    if (inserted) {
      merged.push_back(edge);
      llvm::sort(merged.back().payloads, [](AsyncOp a, AsyncOp b) {
        return static_cast<int>(a) < static_cast<int>(b);
      });
      llvm::sort(merged.back().pieces);
      continue;
    }
    EdgeRec &dst = merged[it->second];
    if (keepLatestSource && followsInChain(edge.src, dst.src))
      dst.src = edge.src;
    absorbEdge(dst, edge);
  }
  return merged;
}

static LogicalResult buildEdgesAndSemas(GroupDag &g, SmallVector<EdgeRec> &edges) {
  if (failed(reduceEdges(g, edges)))
    return failure();
  auto edgeComp = [&](const EdgeRec &e) -> CompId {
    return g.pieceTable.pieceComp[e.pieces.front()];
  };
  auto deduped = mergeEdges(edges, [&](const EdgeRec &edge) {
    return std::make_tuple(edge.src, edge.dst, ownerKey(edge.srcOwner), edgeComp(edge));
  }, false);
  auto collapsed = mergeEdges(deduped, [&](const EdgeRec &edge) {
    return std::make_tuple(edge.dst, ownerKey(edge.srcOwner), edgeComp(edge));
  }, true);
  struct DstGroup {
    Node *dst;
    SmallVector<unsigned, 2> idxs;
    int sema = -1;
  };
  llvm::MapVector<std::tuple<Node *, int64_t, CompId>, unsigned> dstIndex;
  SmallVector<DstGroup> groups;
  for (auto [i, e] : llvm::enumerate(collapsed)) {
    auto key = std::make_tuple(e.dst, ownerKey(e.dstOwner), edgeComp(e));
    auto it = dstIndex.find(key);
    if (it == dstIndex.end()) {
      dstIndex.try_emplace(key, groups.size());
      groups.push_back(DstGroup{e.dst, {static_cast<unsigned>(i)}, -1});
    } else {
      groups[it->second].idxs.push_back(i);
    }
  }
  auto groupComp = [&](const DstGroup &grp) {
    return g.pieceTable.pieceComp[collapsed[grp.idxs.front()].pieces.front()];
  };
  auto groupAcquirer = [&](const DstGroup &grp) {
    return collapsed[grp.idxs.front()].dstOwner;
  };
  auto findRegainGroup = [&](Node *forRow, const Owner &acq, CompId comp) -> int {
    int best = -1;
    for (Node *m = forRow->children[0]; m; m = m->next) {
      auto it = dstIndex.find(std::make_tuple(m, ownerKey(acq), comp));
      if (it == dstIndex.end())
        continue;
      DstGroup &cand = groups[it->second];
      if (groupComp(cand) == comp)
        best = static_cast<int>(it->second);
    }
    return best;
  };
  auto createSema = [&](DstGroup &grp) -> LogicalResult {
    SemaId sid = g.semas.size();
    Sema s;
    s.name = "S" + std::to_string(sid);
    s.count = grp.idxs.size();
    for (unsigned idx : grp.idxs)
      for (PieceId p : collapsed[idx].pieces)
        if (!llvm::is_contained(s.pieces, p))
          s.pieces.push_back(p);
    llvm::sort(s.pieces);
    s.component = g.pieceTable.pieceComp[s.pieces.front()];
    for (PieceId p : s.pieces)
      if (g.pieceTable.pieceComp[p] != s.component)
        return semaError(grp.dst->op ? grp.dst->op : g.root->op)
               << "one destination joins pieces of different components";
    grp.sema = static_cast<int>(sid);
    g.semas.push_back(std::move(s));
    return success();
  };
  for (DstGroup &grp : groups) {
    if (grp.sema != -1)
      continue;
    for (unsigned i = 0; i < grp.idxs.size(); ++i)
      for (unsigned j = i + 1; j < grp.idxs.size(); ++j)
        if (sameOwner(collapsed[grp.idxs[i]].srcOwner, collapsed[grp.idxs[j]].srcOwner))
          return semaError(grp.dst->op ? grp.dst->op : g.root->op)
                 << "fan-in sources share a partition — not expressible as one semaphore";
    if (grp.dst->kind == Node::For) {
      int t = findRegainGroup(grp.dst, groupAcquirer(grp), groupComp(grp));
      if (t >= 0) {
        if (groups[t].sema == -1)
          if (failed(createSema(groups[t])))
            return failure();
        grp.sema = groups[t].sema;
        Sema &s = g.semas[grp.sema];
        for (unsigned idx : grp.idxs)
          for (PieceId p : collapsed[idx].pieces)
            if (!llvm::is_contained(s.pieces, p))
              s.pieces.push_back(p);
        llvm::sort(s.pieces);
        continue;
      }
    }
    if (failed(createSema(grp)))
      return failure();
  }
  DenseMap<Node *, Node *> lastAfter; // release insertion cursor per source
  for (DstGroup &grp : groups) {
    Sema &s = g.semas[grp.sema];
    unsigned m = grp.idxs.size();
    unsigned relCount = 1;
    if (m != s.count) {
      if (m == 1)
        relCount = s.count;
      else
        return semaError(grp.dst->op ? grp.dst->op : g.root->op) << "destination group with "
               << m << " sources cannot meet semaphore " << s.name << " pending count " << s.count;
    }
    Node *acq = newProtocolNode(g, Node::Acquire, grp.dst->parent, groupAcquirer(grp), grp.sema, s.count);
    Node *dstAnchor = grp.dst;
    if (grp.dst->kind == Node::Exit && acq->owner) {
      Node *head = grp.dst;
      while (head->prev)
        head = head->prev;
      Owner firstWaveOwner;
      Node *firstTouch = nullptr;
      for (Node *r = head; r; r = r->next) {
        if (r->kind != Node::Access || !r->owner)
          continue;
        if (!firstWaveOwner)
          firstWaveOwner = r->owner;
        if (!firstTouch && sameOwner(r->owner, acq->owner))
          firstTouch = r;
      }
      if (firstWaveOwner && !sameOwner(firstWaveOwner, acq->owner) && firstTouch) {
        dstAnchor = firstTouch;
        s.isEntry = true; // initial permit; no pre-loop entry instance
        s.inheritStamp = acq->owner;
      }
    }
    acq->scheduleAnchor = dstAnchor;
    spliceBefore(acq, dstAnchor);
    s.expectedReleases += m * relCount;
    for (unsigned idx : grp.idxs) {
      EdgeRec &e = collapsed[idx];
      Node *rel = newProtocolNode(g, Node::Release, e.src->parent, e.srcOwner, grp.sema, relCount);
      rel->payloads = e.payloads;
      rel->sat = acq;
      rel->scheduleAnchor = e.src;
      Node *anchor = lastAfter.lookup(e.src);
      spliceAfter(rel, anchor ? anchor : e.src);
      lastAfter[e.src] = rel;
    }
  }
  return success();
}
static bool nodeInvolvesComp(GroupDag &g, Node *n, CompId comp) {
  if (n->kind == Node::Access)
    return touchesComponent(g, n, comp);
  if (n->isRegion())
    for (auto &[p, pi] : n->pieceInfo)
      if (g.pieceTable.pieceComp[p] == comp)
        return true;
  return false;
}
static Node *lastAcquireOfCompInChain(GroupDag &g, Node *head, CompId comp) {
  Node *found = nullptr;
  forEachNode(head, [&](Node *n) {
    if (n->kind == Node::Acquire && getSema(g, n).component == comp)
      found = n;
  });
  return found;
}
static Owner firstAccessOwnerOfComp(GroupDag &g, Node *head, CompId comp, bool &found) {
  Owner owner;
  forEachNode(head, [&](Node *n) {
    if (!found && n->kind == Node::Access && touchesComponent(g, n, comp)) {
      found = true;
      owner = n->owner;
    }
  });
  return owner;
}

static LogicalResult insertEntryAcquires(GroupDag &g) {
  unsigned numComps = numComponents(g);
  Node *top = g.root->children.empty() ? nullptr : g.root->children[0];
  if (!top)
    return success();
  for (CompId comp = 0; comp < numComps; ++comp) {
    bool hasSync = false;
    for (const Sema &s : g.semas)
      if (s.component == comp)
        hasSync = true;
    if (!hasSync)
      continue;
    Node *chainHead = top;
    SmallVector<Node *, 4> rows;
    auto collectRows = [&](Node *head) {
      rows.clear();
      for (Node *n = head; n; n = n->next)
        if (nodeInvolvesComp(g, n, comp))
          rows.push_back(n);
    };
    collectRows(chainHead);
    while (rows.size() == 1 && rows[0]->kind == Node::If) {
      Node *onlyChild = nullptr;
      int cnt = 0;
      for (Node *child : rows[0]->children) {
        bool involves = false;
        for (Node *n = child; n; n = n->next)
          if (nodeInvolvesComp(g, n, comp))
            involves = true;
        if (involves) {
          onlyChild = child;
          ++cnt;
        }
      }
      if (cnt != 1)
        break;
      chainHead = onlyChild;
      collectRows(chainHead);
    }
    if (rows.empty())
      return semaError(g.root->op) << "component with sync but no placement rows";
    bool fhFound = false;
    Owner firstHolder = firstAccessOwnerOfComp(g, top, comp, fhFound);
    if (!fhFound)
      return semaError(g.root->op) << "component has no access rows";
    Node *regain = nullptr;
    for (Node *row : llvm::reverse(rows))
      if (row->kind == Node::For) {
        regain = lastAcquireOfCompInChain(g, row->children[0], comp);
        if (regain)
          break;
      }
    if (regain) {
      Sema &s = getSema(g, regain);
      s.isEntry = true; // first event in chain order is an acquire
      s.inheritStamp = firstHolder; // carrier inherit: emission stamps this
      Node *acq = newProtocolNode(g, Node::Acquire, rows.front()->parent,
                                  std::nullopt, regain->sema, regain->count);
      spliceBefore(acq, rows.front());
    } else {
      SemaId sid = g.semas.size();
      Sema s;
      s.name = "E" + std::to_string(sid);
      s.component = comp;
      for (auto [p, c] : llvm::enumerate(g.pieceTable.pieceComp))
        if (c == comp)
          s.pieces.push_back(p);
      s.count = 1;
      s.isEntry = true;
      s.expectedReleases = 1; // the terminal release
      s.inheritStamp = firstHolder; // carrier inherit: emission stamps this
      Node *acq = newProtocolNode(g, Node::Acquire, rows.front()->parent, std::nullopt, sid, 1);
      spliceBefore(acq, rows.front());
      Node *terminal = rows.back();
      Owner owner = terminal->kind == Node::Access ? terminal->owner
                        : sortedPieceInfo(terminal).front().second.owner;
      Node *rel = newProtocolNode(g, Node::Release, terminal->parent, owner, sid, 1);
      rel->payloads.push_back(AsyncOp::NONE);
      Node *anchor = terminal;
      while (anchor->next && anchor->next->kind == Node::Release)
        anchor = anchor->next;
      spliceAfter(rel, anchor);
      g.semas.push_back(std::move(s));
    }
  }
  return success();
}
static Node *chainFinalForComp(GroupDag &g, Node *head, CompId comp) {
  Node *final = nullptr;
  for (Node *n = head; n; n = n->next) {
    if (n->kind == Node::Acquire && getSema(g, n).component == comp)
      final = n;
    if (n->isRegion())
      for (const Crossing &c : n->crossings)
        if (c.comp == comp)
          final = n;
  }
  return final;
}
static Owner finalOwner(GroupDag &g, Node *final, CompId comp) {
  if (final->kind == Node::Acquire)
    return final->owner;
  for (const Crossing &c : final->crossings)
    if (c.comp == comp)
      return c.slotOwner;
  return std::nullopt;
}

static void computeCrossings(GroupDag &g, Node *head, unsigned numComps) {
  forEachRegionPostOrder(head, [&](Node *n) {
    for (CompId comp = 0; comp < numComps; ++comp) {
      Crossing cr;
      cr.comp = comp;
      bool any = false;
      for (Node *child : n->children) {
        Node *f = chainFinalForComp(g, child, comp);
        cr.finals.push_back(f);
        if (f) {
          any = true;
          cr.slotOwner = finalOwner(g, f, comp);
        }
      }
      if (any)
        n->crossings.push_back(std::move(cr));
    }
  });
}
static bool carrierConsumedAfter(GroupDag &g, Node *start, CompId comp) {
  for (Node *n = start; n; n = n->next) {
    switch (n->kind) {
    case Node::Acquire:
      if (getSema(g, n).component == comp)
        return false; // fresh carrier supersedes the escaped one
      break;
    case Node::Release:
      if (getSema(g, n).component == comp)
        return true;
      break;
    case Node::Access:
      for (const Touch &t : n->touches)
        if (compOfMember(g, t.member) == comp)
          return true;
      break;
    case Node::For:
    case Node::If:
      for (const Crossing &c : n->crossings)
        if (c.comp == comp)
          return true;
      break;
    default:
      break;
    }
  }
  return false;
}
static bool regionLiveFor(const Node *region, CompId comp) {
  if (!region)
    return false; // function chain
  for (const Crossing &c : region->crossings)
    if (c.comp == comp)
      return true;
  return false;
}

static void pruneDeadIfCrossings(GroupDag &g, Node *head, Node *region) {
  SmallVector<Node *, 8> rows;
  for (Node *n = head; n; n = n->next)
    rows.push_back(n);
  for (Node *n : llvm::reverse(rows))
    if (n->kind == Node::If)
      llvm::erase_if(n->crossings, [&](const Crossing &c) {
        return !carrierConsumedAfter(g, n->next, c.comp) && !regionLiveFor(region, c.comp);
      });
  for (Node *n : rows)
    if (n->isRegion())
      for (Node *child : n->children)
        pruneDeadIfCrossings(g, child, n);
}
static void collectParts(Node *head, SmallVector<int, 4> &parts) {
  forEachNode(head, [&](Node *n) {
    if ((n->kind == Node::Access || n->kind == Node::Acquire || n->kind == Node::Release) &&
        n->owner.has_value())
      if (!llvm::is_contained(parts, n->owner->first))
        parts.push_back(n->owner->first);
  });
}
static bool crossesComp(const Node *n, CompId comp) {
  return llvm::any_of(n->crossings, [&](const Crossing &x) { return x.comp == comp; });
}
static scf::ForOp outerWSLoop(scf::ForOp loop);
static const Crossing *findCrossing(const Node *n, CompId comp) {
  for (const Crossing &c : n->crossings)
    if (c.comp == comp)
      return &c;
  return nullptr;
}
static bool allEnclosersCanDrop(const Node *F) {
  if (F->op && gpu::hasWarpSpecializeTag(F->op))
    return true;
  for (const Node *p = F->parent; p; p = p->parent) {
    if (p->kind == Node::Func)
      return true;
    if (p->kind == Node::If)
      return false;
    if (p->kind != Node::For)
      continue;
    if (p->op && gpu::hasWarpSpecializeTag(p->op))
      return true;
  }
  return true;
}
static bool regionResultConsumedAfter(GroupDag &g, Node *region, CompId comp) {
  for (Node *m = region->next; m; m = m->next) {
    if (m->kind == Node::Acquire && getSema(g, m).component == comp)
      return false;
    if (m->kind == Node::Release && getSema(g, m).component == comp)
      return true;
    if (m->kind == Node::Access && nodeInvolvesComp(g, m, comp))
      return true;
    if (m->isRegion() && crossesComp(m, comp))
      return true;
  }
  Node *p = region->parent;
  if (!p || !p->isRegion())
    return false;
  for (const Crossing &x : p->crossings)
    if (x.comp == comp && llvm::any_of(x.finals, [&](Node *f) { return f == region; }))
      return regionResultConsumedAfter(g, p, comp);
  return false;
}
static bool prefixRowIsSingleBufferView(Node *F, Node *bufRow) {
  if (!F || !F->op || gpu::hasWarpSpecializeTag(F->op))
    return true;
  if (!bufRow || !bufRow->op)
    return false;
  auto alloc = dyn_cast<nvidia_gpu::TMEMAllocOp>(bufRow->op);
  return !alloc || !alloc.getSrc();
}
static bool isAcquireForComp(GroupDag &g, const Node *n, CompId comp) {
  return n->kind == Node::Acquire && getSema(g, n).component == comp;
}
static bool isReleaseForComp(GroupDag &g, const Node *n, CompId comp) {
  return n->kind == Node::Release && getSema(g, n).component == comp;
}
static bool isAccessForComp(GroupDag &g, Node *n, CompId comp) {
  return n->kind == Node::Access && nodeInvolvesComp(g, n, comp);
}
static bool isRegionCrossingForComp(const Node *n, CompId comp) {
  return n && n->isRegion() && crossesComp(n, comp);
}
static bool rowHasCompEvent(GroupDag &g, Node *n, CompId comp) {
  return isAcquireForComp(g, n, comp) || isReleaseForComp(g, n, comp) || nodeInvolvesComp(g, n, comp);
}
static bool isCarrierEvent(GroupDag &g, Node *n, CompId comp) {
  return isAcquireForComp(g, n, comp) || isReleaseForComp(g, n, comp) ||
         isAccessForComp(g, n, comp) || isRegionCrossingForComp(n, comp);
}
static bool precedesInChain(Node *src, Node *dst) {
  for (Node *n = src->next; n; n = n->next)
    if (n == dst)
      return true;
  return false;
}
static bool regionEntryOwner(GroupDag &g, Node *region, CompId comp, Owner &owner) {
  bool found = false;
  for (auto &[p, pi] : sortedPieceInfo(region)) {
    if (g.pieceTable.pieceComp[p] != comp)
      continue;
    if (!found) {
      owner = pi.owner;
      found = true;
      continue;
    }
    if (!sameOwner(owner, pi.owner))
      return false;
  }
  return found;
}
static bool chainHasCompEvent(GroupDag &g, Node *head, CompId comp) {
  for (Node *n = head; n; n = n->next)
    if (rowHasCompEvent(g, n, comp))
      return true;
  return false;
}
static Owner returnedOwnerForFinal(GroupDag &g, Node *final, CompId comp, Owner incoming) {
  if (!final)
    return incoming;
  if (final->kind == Node::Acquire)
    return final->owner;
  if (final->isRegion())
    if (const Crossing *c = findCrossing(final, comp))
      return c->slotOwner;
  return std::nullopt;
}
static std::optional<SemaId>
returnedSemaForFinal(GroupDag &g, Node *final, CompId comp, SemaId incoming) {
  if (!final)
    return incoming;
  if (final->kind == Node::Acquire)
    return final->sema;
  if (!final->isRegion())
    return std::nullopt;
  const Crossing *c = findCrossing(final, comp);
  if (!c)
    return std::nullopt;
  std::optional<SemaId> common;
  if (final->kind == Node::For)
    common = incoming; // zero-trip path.
  for (unsigned i = 0, e = final->children.size(); i < e; ++i) {
    Node *childFinal = i < c->finals.size() ? c->finals[i] : nullptr;
    std::optional<SemaId> child = returnedSemaForFinal(g, childFinal, comp, incoming);
    if (!child)
      return std::nullopt;
    if (!common)
      common = child;
    else if (*common != *child)
      return std::nullopt;
  }
  return common;
}
static bool isHoldTransparentRegion(GroupDag &g, Node *region, CompId comp, Owner holdOwner) {
  const Crossing *rc = findCrossing(region, comp);
  if (!rc)
    return false;
  Owner entryOwner;
  if (!regionEntryOwner(g, region, comp, entryOwner) || !sameOwner(entryOwner, holdOwner))
    return false;
  if (rc->finals.size() > region->children.size())
    return false;
  for (unsigned i = 0, e = region->children.size(); i < e; ++i) {
    Node *childFinal = i < rc->finals.size() ? rc->finals[i] : nullptr;
    if (!childFinal && chainHasCompEvent(g, region->children[i], comp))
      return false;
    if (childFinal && chainHasCompEvent(g, childFinal->next, comp))
      return false;
    Owner returned = returnedOwnerForFinal(g, childFinal, comp, entryOwner);
    if (!sameOwner(returned, holdOwner))
      return false;
  }
  return true;
}
static Hold carrierHold(const char *reason, Node *regain = nullptr) {
  Hold h;
  h.outcome = Hold::Outcome::CARRIER;
  h.reason = reason;
  h.regain = regain;
  return h;
}
static Hold childOwnsHold(Node *regain) {
  Hold h;
  h.outcome = Hold::Outcome::CHILD_OWNS;
  h.regain = regain;
  return h;
}
static Hold pointOfUseHold(Node *firstToucher, Node *entryAcquire, Node *closingRelease, Node *regain,
                           SmallVector<Node *, 4> rows, bool regionTail, bool needsFinalAcquire,
                           Node *finalAcquire, bool keepsEntryAcquire,
                           Node *bridgeAcquire, Node *bridgeRelease) {
  Hold h;
  h.outcome = Hold::Outcome::POINT_OF_USE;
  h.rows = std::move(rows);
  h.entryAcquire = entryAcquire;
  h.closingRelease = closingRelease;
  h.regain = regain;
  h.firstToucher = firstToucher;
  h.finalAcquire = finalAcquire;
  h.needsFinalAcquire = needsFinalAcquire;
  h.keepsEntryAcquire = keepsEntryAcquire;
  h.bridgeAcquire = bridgeAcquire;
  h.bridgeRelease = bridgeRelease;
  h.regionTail = regionTail;
  return h;
}
static bool childOwnsCarrier(const Node *region, CompId comp) {
  if (const Crossing *child = findCrossing(region, comp))
    return !child->hold.materializesCarrier();
  return false;
}
static bool hasTrailingCompUse(GroupDag &g, Node *regain, CompId comp) {
  for (Node *m = regain->next; m; m = m->next)
    if (isAccessForComp(g, m, comp) || isRegionCrossingForComp(m, comp) || isReleaseForComp(g, m, comp))
      return true;
  return false;
}
static Node *findBridgeAcquireAfter(GroupDag &g, Node *F, SemaId feedSema, CompId comp, Owner owner,
                                    Node *existingBridgeRelease) {
  for (Node *m = F->next; m; m = m->next) {
    if (!isAcquireForComp(g, m, comp) || m->sema != feedSema || !sameOwner(m->owner, owner))
      continue;
    for (Node *tail = m->next; tail; tail = tail->next)
      if (isCarrierEvent(g, tail, comp) && tail != existingBridgeRelease)
        return nullptr;
    return m;
  }
  return nullptr;
}

struct HoldFeed {
  Node *acquire = nullptr;
  const char *rejectReason = nullptr;
};
static HoldFeed findHoldFeedAcquire(GroupDag &g, Node *F, CompId comp) {
  for (Node *cur = F;; cur = cur->parent) {
    for (Node *m = cur->prev; m; m = m->prev) {
      if (isAcquireForComp(g, m, comp))
        return {m, nullptr};
      if (isAccessForComp(g, m, comp))
        return {nullptr, "entry-consumed"};
      if (isRegionCrossingForComp(m, comp))
        return {nullptr, "region-feed"};
      if (isReleaseForComp(g, m, comp))
        return {nullptr, "release-feed"};
    }
    if (!cur->parent || !cur->parent->isRegion())
      return {nullptr, "no-entry-acquire"};
  }
}

struct HoldPrefix {
  Node *firstToucher = nullptr;
  Node *closingRelease = nullptr;
  SmallVector<Node *, 4> rows;
  unsigned releases = 0;
  bool releaseBeforeFirstToucher = false;
  const char *rejectReason = nullptr;
};
static HoldPrefix analyzeHoldPrefix(GroupDag &g, Node *F, CompId comp,
                                    Owner slotOwner, Node *regain, bool regionTail) {
  HoldPrefix p;
  for (Node *m = F->children[0]; m; m = m->next) {
    if (regionTail && m == regain)
      break;
    if (isAcquireForComp(g, m, comp))
      break;
    if (isRegionCrossingForComp(m, comp)) {
      if (!p.firstToucher) {
        p.rejectReason = "region-crossing";
        return p;
      }
      if (!isHoldTransparentRegion(g, m, comp, slotOwner)) {
        p.rejectReason = "region-not-transparent";
        return p;
      }
      p.rows.push_back(m);
      continue;
    }
    if (isAccessForComp(g, m, comp)) {
      if (!p.firstToucher) {
        p.firstToucher = m;
        p.releaseBeforeFirstToucher = p.releases != 0;
      }
      p.rows.push_back(m);
    }
    if (isReleaseForComp(g, m, comp)) {
      p.releases += std::max(1u, m->count);
      if (!p.closingRelease)
        p.closingRelease = m;
    }
  }
  unsigned expectedReleases = regionTail ? 0 : 1;
  if (!p.firstToucher)
    p.rejectReason = "no-buf";
  else if (p.releases != expectedReleases)
    p.rejectReason = "rel-count";
  else if (p.releaseBeforeFirstToucher)
    p.rejectReason = "rel-before-buf";
  return p;
}
// Select carrier, point-of-use, or child-owned realization for one loop crossing.
static Hold buildUniformHold(GroupDag &g, Node *F, const Crossing &c) {
  auto forOp = F->op ? dyn_cast<scf::ForOp>(F->op) : scf::ForOp();
  if (!forOp || !gpu::hasWarpSpecializeTag(outerWSLoop(forOp)))
    return carrierHold("non-ws-scope");
  if (!allEnclosersCanDrop(F))
    return carrierHold("if-encloser");
  CompId comp = c.comp;
  Node *regain = c.finals.empty() ? nullptr : c.finals[0];
  bool regionTail = false;
  if (!regain)
    return carrierHold("no-final");
  if (regain->finalPermissionAcquire)
    return carrierHold("final-permission", regain);
  if (regain->kind == Node::For && childOwnsCarrier(regain, comp))
    return childOwnsHold(regain);
  if (regain->isRegion()) {
    if (!isHoldTransparentRegion(g, regain, comp, c.slotOwner))
      return carrierHold("region-not-transparent", regain);
    regionTail = true;
  } else if (regain->kind != Node::Acquire) {
    return carrierHold("region-not-transparent", regain);
  }
  if (hasTrailingCompUse(g, regain, comp))
    return carrierHold("trailing-use", regain);
  HoldFeed feed = findHoldFeedAcquire(g, F, comp);
  if (!feed.acquire)
    return carrierHold(feed.rejectReason, regain);
  std::optional<SemaId> regainSema = regionTail ? returnedSemaForFinal(g, regain, comp, feed.acquire->sema)
                 : std::optional<SemaId>(regain->sema);
  bool parentConsumesResult = regionResultConsumedAfter(g, F, comp);
  bool needsFinalAcquire = c.hold.finalAcquire || parentConsumesResult;
  if (!regainSema)
    return carrierHold("entry-sema-mismatch", regain);
  bool keepsEntryAcquire = feed.acquire->sema != *regainSema;
  Node *bridgeAcquire = c.hold.bridgeAcquire;
  if (!keepsEntryAcquire && needsFinalAcquire)
    return carrierHold("result-consumed", regain);
  if (keepsEntryAcquire) {
    const Sema &feedSema = getSema(g, feed.acquire);
    Owner feedOwner = feed.acquire->owner ? feed.acquire->owner : feedSema.inheritStamp;
    if (!bridgeAcquire)
      bridgeAcquire = findBridgeAcquireAfter(
          g, F, feed.acquire->sema, comp, c.slotOwner, c.hold.bridgeRelease);
    if (regionTail || !needsFinalAcquire || !sameOwner(feedOwner, c.slotOwner) || !bridgeAcquire)
      return carrierHold("entry-sema-mismatch", regain);
  }
  HoldPrefix prefix = analyzeHoldPrefix(g, F, comp, c.slotOwner, regain, regionTail);
  if (prefix.rejectReason)
    return carrierHold(prefix.rejectReason, regain);
  if (!prefixRowIsSingleBufferView(F, prefix.firstToucher))
    return carrierHold("prefix-not-buffer-view", regain);
  return pointOfUseHold(prefix.firstToucher, feed.acquire,
                        prefix.closingRelease, regain, std::move(prefix.rows),
                        regionTail, needsFinalAcquire, c.hold.finalAcquire,
                        keepsEntryAcquire, bridgeAcquire, c.hold.bridgeRelease);
}

static LogicalResult computeHoldRules(GroupDag &g, Node *head) {
  LogicalResult result = success();
  forEachRegionPostOrder(head, [&](Node *n) {
    if (failed(result))
      return;
    if (n->kind == Node::For) {
      for (Crossing &c : n->crossings) {
        Node *finalAcquire = c.hold.finalAcquire;
        Hold next = buildUniformHold(g, n, c);
        if (finalAcquire && (!next.isPointOfUse() || next.finalAcquire != finalAcquire)) {
          result = semaError(n->op) << "final-permission acquire invalidated its point-of-use hold";
          return;
        }
        c.hold = std::move(next);
      }
    }
  });
  return result;
}
static void unlinkFromChain(Node *n) {
  if (n->prev)
    n->prev->next = n->next;
  else if (n->parent)
    for (Node *&child : n->parent->children)
      if (child == n)
        child = n->next;
  if (n->next)
    n->next->prev = n->prev;
  n->prev = n->next = nullptr;
}

static bool materializeFinalPermissionAcquires(GroupDag &g, Node *head) {
  bool changed = false;
  forEachRegionPostOrder(head, [&](Node *n) {
    if (n->kind != Node::For)
      return;
    Node *anchor = n;
    for (Crossing &c : n->crossings) {
      if (!c.hold.isPointOfUse() || !c.hold.needsFinalAcquire)
        continue;
      Node *recurrenceAcquire = c.hold.regionTail ? c.hold.entryAcquire : c.hold.regain;
      if (!c.hold.finalAcquire) {
        Node *finalAcquire = newProtocolNode(
            g, Node::Acquire, n->parent, c.slotOwner, recurrenceAcquire->sema, recurrenceAcquire->count);
        finalAcquire->finalPermissionAcquire = true;
        spliceAfter(finalAcquire, anchor);
        anchor = finalAcquire;
        c.hold.finalAcquire = finalAcquire;
        changed = true;
      } else {
        anchor = c.hold.finalAcquire;
      }
      if (c.hold.keepsEntryAcquire && !c.hold.bridgeRelease) {
        Node *bridge = newProtocolNode(g, Node::Release, c.hold.bridgeAcquire->parent, c.slotOwner,
            recurrenceAcquire->sema, recurrenceAcquire->count);
        bridge->payloads.push_back(AsyncOp::NONE);
        spliceAfter(bridge, c.hold.bridgeAcquire);
        getSema(g, bridge).expectedReleases += bridge->count;
        c.hold.bridgeRelease = bridge;
        changed = true;
      }
    }
  });
  return changed;
}
static void refreshCrossingFinals(GroupDag &g, Node *head) {
  forEachRegionPostOrder(head, [&](Node *n) {
    for (Crossing &c : n->crossings) {
      c.finals.clear();
      for (Node *child : n->children) {
        Node *final = chainFinalForComp(g, child, c.comp);
        c.finals.push_back(final);
        if (final)
          c.slotOwner = finalOwner(g, final, c.comp);
      }
    }
  });
}

static void applyHoldRulePlacement(GroupDag &g, Node *head) {
  forEachRegionPostOrder(head, [&](Node *n) {
    if (n->kind != Node::For)
      return;
    for (Crossing &c : n->crossings) {
      if (!c.hold.isPointOfUse())
        continue;
      if (c.hold.regionTail) {
        Node *tail = c.finals[0];
        Node *pointAcquire = c.hold.entryAcquire;
        Sema &s = getSema(g, pointAcquire);
        pointAcquire->owner = c.slotOwner;
        unlinkFromChain(pointAcquire);
        pointAcquire->scheduleAnchor = c.hold.firstToucher;
        spliceBefore(pointAcquire, c.hold.firstToucher);
        Node *closing = newProtocolNode(g, Node::Release, tail->parent, c.slotOwner, pointAcquire->sema, 1);
        closing->payloads.push_back(AsyncOp::NONE);
        closing->sat = pointAcquire;
        closing->scheduleAnchor = tail;
        spliceAfter(closing, tail);
        s.expectedReleases += closing->count;
        c.hold.closingRelease = closing;
        continue;
      }
      Node *regain = c.finals[0];
      unlinkFromChain(regain);
      regain->scheduleAnchor = c.hold.firstToucher;
      spliceBefore(regain, c.hold.firstToucher);
      if (c.hold.keepsEntryAcquire) {
        Sema &recurrenceSema = getSema(g, regain);
        recurrenceSema.isEntry = true;
        recurrenceSema.inheritStamp = c.slotOwner;
      } else {
        unlinkFromChain(c.hold.entryAcquire);
      }
    }
  });
}
static void computeRequiredParts(Node *head) {
  forEachNode(head, [&](Node *n) {
    if (n->isRegion()) {
      SmallVector<int, 4> parts;
      for (Node *child : n->children)
        collectParts(child, parts);
      llvm::sort(parts);
      n->requiredParts.assign(parts.begin(), parts.end());
    }
  });
}
static bool canDoubleBufferAcc(nvidia_gpu::MMAv5OpInterface mmaOp, int numTmemBlocks) {
  auto tmemDesc = mmaOp.getAccumulator().getType();
  int64_t blockM = tmemDesc.getShape()[0];
  int64_t blockN = tmemDesc.getShape()[1];
  constexpr int numTMEMColumns = 512;
  constexpr int numTMEMRows = 128;
  if (numTmemBlocks + blockM * blockN * 2 > numTMEMRows * numTMEMColumns)
    return false;
  if (isa<nvidia_gpu::TCGen5MMAScaledOp>(mmaOp.getOperation()) && blockN == 256)
    return false;
  return true;
}
static scf::ForOp outerWSLoop(scf::ForOp loop) {
  scf::ForOp ws = loop;
  for (Operation *p = loop; p; p = p->getParentOp())
    if (auto f = dyn_cast<scf::ForOp>(p))
      if (gpu::hasWarpSpecializeTag(f))
        ws = f;
  return ws;
}
static bool isMultiStagedGroup(GroupDag &g, int numTmemBlocks) {
  bool isMultiStaged = true;
  for (const Member &m : g.pieceTable.members) {
    for (Operation *user : m.allocOp->getResult(0).getUsers()) {
      if (auto mmaOp = dyn_cast<nvidia_gpu::MMAv5OpInterface>(user)) {
        if (auto loop = dyn_cast<scf::ForOp>(user->getParentOp())) {
          scf::ForOp wsLoop = outerWSLoop(loop);
          bool accIsMultiBuffered = !nvidia_gpu::hasAccReadModifyWrite(mmaOp, loop) &&
              nvidia_gpu::isAccMultibufferingPossible(mmaOp, loop) &&
              !getDisallowAccMultiBuffer(wsLoop) && canDoubleBufferAcc(mmaOp, numTmemBlocks);
          isMultiStaged = isMultiStaged && accIsMultiBuffered;
        }
      }
    }
  }
  return isMultiStaged;
}
static LogicalResult getPlannedBufferCopy(GroupDag &g, std::optional<int> &plannedCopy) {
  bool sawMissing = false;
  for (const Member &m : g.pieceTable.members) {
    auto copyAttr = m.allocOp->getAttrOfType<IntegerAttr>("buffer.copy");
    if (!copyAttr) {
      sawMissing = true;
      continue;
    }
    int copy = copyAttr.getInt();
    if (copy < 1)
      return semaError(m.allocOp) << "planned buffer.copy must be positive";
    if (plannedCopy && *plannedCopy != copy)
      return semaError(m.allocOp) << "allocs in one planned reuse group have inconsistent buffer.copy values";
    plannedCopy = copy;
  }
  if (plannedCopy && sawMissing)
    return semaError(g.pieceTable.members.front().allocOp)
           << "planned reuse group mixes buffer.copy and non-buffer.copy allocs";
  return success();
}

LogicalResult computeBackingPlan(GroupDag &g, bool useMetaPartitioner, int lowerSemaphoreNumStages,
                                 int &numTmemBlocks) {
  g.numStages = 1;
  bool untouched = g.semas.empty();
  std::optional<int> plannedBufferCopy;
  if (failed(getPlannedBufferCopy(g, plannedBufferCopy)))
    return failure();
  if (!untouched && plannedBufferCopy) {
    g.numStages = *plannedBufferCopy;
  } else if (g.isTmem() && !untouched && !useMetaPartitioner && isMultiStagedGroup(g, numTmemBlocks))
    g.numStages = 2;
  g.semaphoreDepth = g.numStages;
  bool hasProducerLoad = false;
  forEachNode(g, [&](Node *node) {
    if (node->kind == Node::Release && llvm::is_contained(node->payloads, AsyncOp::TMALoad))
      hasProducerLoad = true;
  });
  if (!untouched && g.isLocal() && !plannedBufferCopy && hasProducerLoad)
    g.semaphoreDepth = std::max(1, lowerSemaphoreNumStages);
  if (g.isTmem() && !untouched)
    for (const Member &m : g.pieceTable.members) {
      auto shape = m.type.getShape();
      if (shape.size() >= 2)
        numTmemBlocks += shape[0] * shape[1] * g.numStages;
    }
  return success();
}

static LogicalResult verifyCarrierLocality(GroupDag &g, Node *head) {
  std::map<CompId, Owner> carrier; // tracked comps only
  auto seedFromPieces = [&](Node *n) {
    for (auto &[p, pi] : sortedPieceInfo(n)) {
      CompId c = g.pieceTable.pieceComp[p];
      auto it = carrier.find(c);
      if (it == carrier.end())
        carrier.emplace(c, pi.owner);
      else if (!sameOwner(it->second, pi.owner))
        carrier.erase(it); // mixed: untracked
    }
  };
  if (head->kind == Node::Enter)
    seedFromPieces(head);
  for (Node *n = head; n; n = n->next) {
    switch (n->kind) {
    case Node::Acquire: {
      const Sema &sm = getSema(g, n);
      carrier[sm.component] = sm.isEntry && !n->owner ? sm.inheritStamp : n->owner;
      break;
    }
    case Node::Release: {
      const Sema &sm = getSema(g, n);
      auto it = carrier.find(sm.component);
      if (n->owner && it != carrier.end() && it->second && !sameOwner(it->second, n->owner))
        return semaError(n->sat && n->sat->op ? n->sat->op : g.root->op)
               << "wave-locality violation: release owned by partition "
               << n->owner->first << " consumes a carrier held by partition " << it->second->first;
      break;
    }
    case Node::Access: {
      if (!n->owner)
        break; // root rows: outside partition discipline
      for (const Touch &t : n->touches) {
        CompId c = compOfMember(g, t.member);
        auto it = carrier.find(c);
        if (it != carrier.end() && it->second && !sameOwner(it->second, n->owner))
          return semaError(n->op) << "wave-locality violation: access owned by partition "
                 << n->owner->first << " touches a carrier held by partition " << it->second->first;
      }
      break;
    }
    case Node::For:
    case Node::If: {
      for (Node *child : n->children)
        if (failed(verifyCarrierLocality(g, child)))
          return failure();
      seedFromPieces(n); // EXIT regain: carrier back with carried owner
      break;
    }
    default:
      break;
    }
  }
  if (head->parent && head->parent->kind == Node::For) {
    auto carrierConsumed = [&](CompId comp) {
      for (Node *n = head; n; n = n->next) {
        if (n->isProtocol() && getSema(g, n).component == comp)
          return n->kind == Node::Release;
        if ((n->kind == Node::Access && touchesComponent(g, n, comp)) ||
            (n->isRegion() && crossesComp(n, comp)))
          return true;
      }
      return false;
    };
    DenseSet<CompId> checked;
    for (const Crossing &c : head->parent->crossings)
      if (c.hold.materializesCarrier() && carrierConsumed(c.comp))
        checked.insert(c.comp);
    auto rowCarried = [&](Node *n) {
      if (n->isProtocol())
        return checked.contains(getSema(g, n).component);
      if (n->kind == Node::Access)
        for (const Touch &t : n->touches)
          if (checked.contains(compOfMember(g, t.member)))
            return true;
      return false;
    };
    std::function<Owner(Node *)> firstWaveOf = [&](Node *h) -> Owner {
      for (Node *n = h; n; n = n->next) {
        if ((n->kind == Node::Access || n->kind == Node::Acquire) && n->owner && rowCarried(n))
          return n->owner;
        if (n->isRegion() && !n->children.empty())
          if (Owner o = firstWaveOf(n->children[0]))
            return o;
      }
      return std::nullopt;
    };
    Owner firstWave = firstWaveOf(head);
    Owner finalCarrier;
    for (Node *n = head; n; n = n->next)
      if (n->kind == Node::Acquire && n->owner && rowCarried(n))
        finalCarrier = n->owner;
    if (firstWave && finalCarrier && !sameOwner(firstWave, finalCarrier))
      return semaError(g.root->op ? g.root->op : head->parent->op)
             << "traversal-boundary wave-locality violation: loop body's " "final carrier owner "
             << finalCarrier->first << " differs from its first wave owner " << firstWave->first;
  }
  return success();
}

static LogicalResult verifyPointOfUseTransparency(GroupDag &g, Node *F, const Crossing &c) {
  const Hold &h = c.hold;
  auto errorAt = [&](Node *node) {
    return semaError(node && node->op ? node->op : F->op);
  };
  Node *pointAcquire = h.regionTail ? h.entryAcquire : h.regain;
  Node *recurrenceAcquire = pointAcquire;
  if (pointAcquire->next != h.firstToucher)
    return errorAt(F) << "point-of-use acquire is not adjacent to its first buffer use";
  for (Node *m = F->children[0]; m && m != pointAcquire; m = m->next) {
    if (isCarrierEvent(g, m, c.comp))
      return errorAt(m) << "carrier event before point-of-use acquire";
  }
  auto verifyTransparentRegion = [&](Node *region) -> LogicalResult {
    if (!isHoldTransparentRegion(g, region, c.comp, c.slotOwner))
      return errorAt(region) << "non-transparent region reached point-of-use hold";
    return success();
  };
  bool sawFirstToucher = false;
  for (Node *row : h.rows) {
    if (row == h.firstToucher)
      sawFirstToucher = true;
    if (isAccessForComp(g, row, c.comp))
      continue;
    if (isRegionCrossingForComp(row, c.comp)) {
      if (failed(verifyTransparentRegion(row)))
        return failure();
      continue;
    }
    return errorAt(row) << "invalid row recorded in point-of-use hold";
  }
  if (!sawFirstToucher)
    return errorAt(h.firstToucher) << "point-of-use hold first toucher is not a recorded hold row";
  if (h.regionTail)
    if (failed(verifyTransparentRegion(h.regain)))
      return failure();
  if (h.regionTail) {
    Node *closing = h.regain->next;
    if (!closing || closing->kind != Node::Release || getSema(g, closing).component != c.comp ||
        closing->sema != h.entryAcquire->sema || closing != h.closingRelease)
      return errorAt(F) << "regionTail point-of-use lacks closing release after region result";
  } else if (!h.closingRelease || std::max(1u, h.closingRelease->count) != 1) {
    Operation *op = h.closingRelease && h.closingRelease->op ? h.closingRelease->op : F->op;
    return semaError(op) << "point-of-use hold requires exactly one closing release";
  }
  if (h.needsFinalAcquire) {
    Node *finalAcquire = h.finalAcquire;
    if (!finalAcquire || finalAcquire->kind != Node::Acquire ||
        !finalAcquire->finalPermissionAcquire || finalAcquire->parent != F->parent ||
        finalAcquire->sema != recurrenceAcquire->sema || finalAcquire->count != recurrenceAcquire->count ||
        !sameOwner(finalAcquire->owner, c.slotOwner))
      return errorAt(F) << "malformed final-permission acquire";
    bool reachedFinalAcquire = false;
    for (Node *m = F->next; m; m = m->next) {
      if (m == finalAcquire) {
        reachedFinalAcquire = true;
        break;
      }
      if (isCarrierEvent(g, m, c.comp))
        return errorAt(m) << "component use precedes its final-permission acquire";
    }
    if (!reachedFinalAcquire)
      return errorAt(F) << "final-permission acquire is not after its loop";
  } else if (h.finalAcquire) {
    return errorAt(F) << "unexpected final-permission acquire";
  }
  if (h.keepsEntryAcquire) {
    const Sema &recurrenceSema = getSema(g, recurrenceAcquire);
    Node *bridgeAcquire = h.bridgeAcquire;
    Node *bridgeRelease = h.bridgeRelease;
    if (!recurrenceSema.isEntry || !bridgeAcquire || !bridgeRelease ||
        bridgeAcquire->kind != Node::Acquire || bridgeAcquire->sema != h.entryAcquire->sema ||
        bridgeAcquire->parent != F->parent || !sameOwner(bridgeAcquire->owner, c.slotOwner) ||
        bridgeRelease->kind != Node::Release || bridgeRelease->parent != bridgeAcquire->parent ||
        bridgeRelease->sema != recurrenceAcquire->sema || bridgeRelease->count != recurrenceAcquire->count ||
        !sameOwner(bridgeRelease->owner, c.slotOwner) || bridgeRelease->sat)
      return errorAt(F) << "malformed outer-to-local semaphore bridge";
    if (!precedesInChain(F, bridgeAcquire) || !precedesInChain(bridgeAcquire, bridgeRelease))
      return errorAt(F) << "semaphore bridge is not after its loop";
  } else if (h.bridgeAcquire || h.bridgeRelease) {
    return errorAt(F) << "unexpected outer-to-local semaphore bridge";
  }
  return success();
}

static LogicalResult verifySyncDag(GroupDag &g) {
  if (!g.semas.empty() && !g.root->children.empty())
    if (failed(verifyCarrierLocality(g, g.root->children[0])))
      return failure();
  auto verifyHold = [&](Node *n) -> LogicalResult {
    if (!n->isRegion())
      return success();
    DenseSet<CompId> seen;
    for (const Crossing &c : n->crossings) {
      if (!seen.insert(c.comp).second)
        return semaError(n->op) << "duplicate carrier slot for component " << c.comp;
      if (c.hold.materializesCarrier())
        continue;
      if (c.hold.isPointOfUse()) {
        if (c.finals.empty() || !c.finals[0] || !c.hold.regain ||
            c.hold.regain != c.finals[0] || !c.hold.firstToucher || !c.hold.entryAcquire)
          return semaError(n->op) << "malformed point-of-use hold crossing";
        if (!c.hold.regionTail && c.hold.regain->kind != Node::Acquire)
          return semaError(n->op) << "point-of-use hold without acquire regain";
        if (c.hold.regionTail && !c.hold.regain->isRegion())
          return semaError(n->op) << "regionTail point-of-use without region regain";
        if (failed(verifyPointOfUseTransparency(g, n, c)))
          return failure();
        continue;
      }
      if (c.finals.empty() || !c.finals[0] ||
          c.finals[0]->kind != Node::For || c.hold.firstToucher || c.hold.entryAcquire)
        return semaError(n->op) << "malformed child-owned hold crossing";
      const Crossing *child = findCrossing(c.finals[0], c.comp);
      if (!child || child->hold.materializesCarrier())
        return semaError(n->op) << "child-owned hold without native child";
    }
    return success();
  };
  if (!g.root->children.empty())
    if (failed(forEachNodeChecked(g.root->children[0], verifyHold)))
      return failure();
  SmallVector<unsigned> releaseCount(g.semas.size(), 0);
  SmallVector<std::optional<int64_t>> acqClass(g.semas.size(), std::nullopt);
  auto verifySemaNode = [&](Node *n) -> LogicalResult {
      if (n->kind == Node::Release) {
        releaseCount[n->sema] += std::max(1u, n->count);
        if (n->payloads.empty())
          return semaError(g.root->op) << "release without payload record";
        if (n->sat) {
          if (n->sat->parent != n->parent)
            return semaError(g.root->op) << "release and its acquire are in different chains";
          bool forward = false;
          for (Node *m = n->next; m; m = m->next)
            if (m == n->sat)
              forward = true;
          if (!forward && !getSema(g, n).isEntry)
            return semaError(g.root->op) << "release does not precede its acquire";
        }
      }
      if (n->kind == Node::Acquire && n->count != getSema(g, n).count)
        return semaError(g.root->op) << "semaphore " << getSema(g, n).name
               << " acquired with non-uniform pending count";
      if (n->kind == Node::Acquire && n->owner.has_value()) {
        int64_t k = ownerKey(n->owner);
        if (acqClass[n->sema] && *acqClass[n->sema] != k)
          return semaError(g.root->op) << "semaphore " << getSema(g, n).name
                 << " acquired by two distinct partitions (M3 violation)";
        acqClass[n->sema] = k;
      }
      return success();
  };
  if (!g.root->children.empty())
    if (failed(forEachNodeChecked(g.root->children[0], verifySemaNode)))
      return failure();
  for (auto [sid, s] : llvm::enumerate(g.semas)) {
    if (releaseCount[sid] != s.expectedReleases)
      return semaError(g.root->op) << "semaphore "
             << s.name << " has " << releaseCount[sid] << " releases, expected " << s.expectedReleases;
  }
  return success();
}
using ScheduleCache = DenseMap<int64_t, gpu::StageCluster>;

struct ScheduleEdge {
  Operation *src = nullptr;
  Operation *dst = nullptr;
};

struct SlotSchedule {
  int64_t advancesPerIteration = 0;
  DenseMap<Node *, int64_t> ordinalByAccess;
  bool complete = true;
};
static Effect accessEffect(const Node *n) {
  Effect effect = Effect::R;
  for (const Touch &touch : n->touches)
    effect = joinEffect(effect, touch.effect);
  return effect;
}

// Derive physical ring displacements before EMIT so protocol and views agree.
static LogicalResult assignCircularStageOffsets(MutableArrayRef<GroupDag> groups) {
  llvm::MapVector<int64_t, SmallVector<GroupDag *, 4>> sets;
  for (GroupDag &g : groups)
    if (g.isCircular() && !g.semas.empty())
      sets[g.bufferId].push_back(&g);
  struct Event {
    GroupDag *group;
    Node *access;
  };
  for (auto &[id, set] : sets) {
    (void)id;
    auto type = set.front()->pieceTable.members.front().type;
    int64_t depth = set.front()->numStages;
    DenseSet<int64_t> starts;
    DenseMap<Operation *, SmallVector<Event, 1>> eventsByOp;
    for (GroupDag *g : set) {
      const Member &member = g->pieceTable.members.front();
      if (!g->isLocal() || g->pieceTable.members.size() != 1)
        return semaError(g->root->op) << "malformed circular local logical group";
      if (member.type != type)
        return semaError(member.allocOp) << "circular local group has mismatched member types";
      if (g->numStages != depth)
        return semaError(member.allocOp) << "circular local group has mismatched depth";
      if (member.circularStart < 0 || member.circularStart >= depth)
        return semaError(member.allocOp) << "circular buffer.start is outside buffer.copy";
      if (!starts.insert(member.circularStart).second)
        return semaError(member.allocOp) << "duplicate circular buffer.start in one group";
      forEachNode(*g, [&](Node *n) {
        if (n->kind == Node::Access)
          eventsByOp[n->op].push_back(Event{g, n});
      });
    }
    SmallVector<Event> ordered;
    cast<triton::FuncOp>(set.front()->root->op).walk([&](Operation *op) {
      if (auto it = eventsByOp.find(op); it != eventsByOp.end())
        ordered.append(it->second.begin(), it->second.end());
    });
    int64_t ordinal = -1;
    DenseMap<GroupDag *, int64_t> lastProduced;
    DenseMap<Node *, int64_t> offset;
    for (Event event : ordered) {
      const Member &member = event.group->pieceTable.members.front();
      if (accessEffect(event.access) == Effect::W) {
        ++ordinal;
        if (member.circularStart != ordinal % depth)
          return semaError(member.allocOp) << "circular producer order expects buffer.start "
                 << ordinal % depth << ", got " << member.circularStart;
        lastProduced[event.group] = ordinal;
        offset[event.access] = 0;
      } else {
        auto it = lastProduced.find(event.group);
        if (it == lastProduced.end())
          return semaError(member.allocOp) << "circular consumer appears before producer";
        offset[event.access] = it->second - ordinal;
      }
      event.access->bufferStageOffset = offset.lookup(event.access);
    }
    for (GroupDag *g : set)
      forEachNode(*g, [&](Node *n) {
        Node *access;
        if (n->kind == Node::Acquire) {
          access = n;
          while ((access = access->next))
            if (access->kind == Node::Access && offset.contains(access))
              break;
        } else if (n->kind == Node::Release) {
          access = n;
          while ((access = access->prev))
            if (access->kind == Node::Access && offset.contains(access))
              break;
        } else {
          return;
        }
        if (access)
          n->stageOffset = offset.lookup(access);
      });
  }
  return success();
}

struct MixedDepthLifecycle {
  GroupDag *group = nullptr;
  Node *writer = nullptr;
  Node *reader = nullptr;
};
static LogicalResult collectMixedDepthLifecycle(GroupDag &group, MixedDepthLifecycle &life) {
  life.group = &group;
  LogicalResult result = success();
  forEachNode(group, [&](Node *node) {
    if (failed(result) || node->kind != Node::Access)
      return;
    Node *&slot = accessEffect(node) == Effect::W ? life.writer : life.reader;
    if (slot) {
      result = semaError(node->op) << "mixed-depth TMEM reuse requires exactly one writer and "
                  "one reader per logical channel";
      return;
    }
    slot = node;
  });
  if (failed(result))
    return failure();
  if (!life.writer || !life.reader)
    return semaError(group.pieceTable.members.front().allocOp)
           << "mixed-depth TMEM reuse requires exactly one writer and one " "reader per logical channel";
  return success();
}

static LogicalResult addMixedDepthAliasScheduleEdges(MutableArrayRef<GroupDag> groups,
    llvm::MapVector<Operation *, SmallVector<ScheduleEdge, 4>> &edgesByLoop) {
  llvm::MapVector<int64_t, SmallVector<GroupDag *, 2>> sets;
  for (GroupDag &group : groups)
    if (group.mixedDepthPhysicalAlias)
      sets[group.bufferId].push_back(&group);
  for (auto &[bufferId, set] : sets) {
    if (set.size() != 2)
      return semaError(set.front()->pieceTable.members.front().allocOp)
             << "mixed-depth TMEM reuse requires exactly two logical channels " "for buffer.id " << bufferId;
    for (GroupDag *group : set) {
      if (group->pieceTable.members.size() != 1 || !group->isTmem())
        return semaError(group->root->op) << "malformed mixed-depth TMEM logical group";
    }
    bool firstOwns = canOwnMixedDepthTmem(*set[0], *set[1]);
    bool secondOwns = canOwnMixedDepthTmem(*set[1], *set[0]);
    if (firstOwns == secondOwns)
      return semaError(set.front()->pieceTable.members.front().allocOp)
             << "mixed-depth TMEM reuse has no unique physical owner by span " "and element width";
    GroupDag *owner = firstOwns ? set[0] : set[1];
    GroupDag *reuser = firstOwns ? set[1] : set[0];
    if (owner->numStages == reuser->numStages)
      return semaError(reuser->pieceTable.members.front().allocOp)
             << "mixed-depth TMEM reuse lost its distinct logical copy depths";
    MixedDepthLifecycle a, b;
    if (failed(collectMixedDepthLifecycle(*owner, a)) || failed(collectMixedDepthLifecycle(*reuser, b)))
      return failure();
    if (!a.writer->owner || !a.reader->owner || !b.writer->owner ||
        !b.reader->owner || a.writer->owner != b.reader->owner ||
        a.reader->owner != b.writer->owner || a.writer->owner == a.reader->owner)
      return semaError(reuser->pieceTable.members.front().allocOp)
             << "mixed-depth TMEM reuse does not form the required alternating " "two-owner cycle";
    Operation *ops[] = {a.writer->op, a.reader->op, b.writer->op, b.reader->op};
    Block *block = ops[0]->getBlock();
    if (llvm::any_of(ArrayRef<Operation *>(ops), [&](Operation *op) { return op->getBlock() != block; }) ||
        !a.writer->op->isBeforeInBlock(b.reader->op) || !a.reader->op->isBeforeInBlock(b.writer->op))
      return semaError(reuser->pieceTable.members.front().allocOp)
             << "mixed-depth TMEM reuse is not ordered as owner.write -> "
                "reuser.read and owner.read -> reuser.write";
    auto loop = a.writer->op->getParentOfType<scf::ForOp>();
    if (!loop || !loop->hasAttr(triton::kScheduledMaxStageAttrName))
      return semaError(a.writer->op) << "mixed-depth TMEM reuse requires one scheduled loop";
    for (Operation *op : ops)
      if (loop.getBody()->findAncestorOpInBlock(*op) != op)
        return semaError(op) << "mixed-depth TMEM reuse accesses must be direct operations "
                  "in one scheduled loop body";
    gpu::StageCluster ownerRead = gpu::getStageCluster(a.reader->op);
    gpu::StageCluster reuserWrite = gpu::getStageCluster(b.writer->op);
    gpu::StageCluster reuserRead = gpu::getStageCluster(b.reader->op);
    gpu::StageCluster ownerWrite = gpu::getStageCluster(a.writer->op);
    if (!ownerRead || !reuserWrite || !reuserRead || !ownerWrite)
      return semaError(a.writer->op) << "mixed-depth TMEM reuse accesses require fixed "
                "loop.stage/loop.cluster annotations";
    int64_t sameIterationSlack = reuserWrite->first - ownerRead->first;
    if (sameIterationSlack < 0)
      return semaError(a.reader->op) << "mixed-depth TMEM same-iteration handoff has negative "
                "pipeline-stage slack";
    if (sameIterationSlack == 0)
      edgesByLoop[loop.getOperation()].push_back(ScheduleEdge{a.reader->op, b.writer->op});
    int64_t backedgeSlack = 1 + ownerWrite->first - reuserRead->first;
    if (backedgeSlack <= 0)
      return semaError(b.reader->op) << "mixed-depth TMEM backedge does not have positive "
                "pipeline-stage slack";
  }
  return success();
}
static Operation *realScheduleAnchor(Node *anchor, bool source) {
  for (Node *n = anchor; n; n = source ? n->prev : n->next) {
    if (n->kind == Node::Access)
      return source && n->completionAnchor ? n->completionAnchor : n->op;
    if (n->isRegion() && n->op)
      return n->op;
  }
  return nullptr;
}

struct LoopAnchorPair {
  scf::ForOp loop;
  Operation *src = nullptr;
  Operation *dst = nullptr;
};
static std::optional<LoopAnchorPair>
findCommonScheduledLoop(Operation *src, Operation *dst) {
  for (Operation *parent = src->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (!loop || !loop->hasAttr(triton::kScheduledMaxStageAttrName))
      continue;
    Operation *srcInLoop = loop.getBody()->findAncestorOpInBlock(*src);
    Operation *dstInLoop = loop.getBody()->findAncestorOpInBlock(*dst);
    if (srcInLoop && dstInLoop)
      return LoopAnchorPair{loop, srcInLoop, dstInLoop};
  }
  return std::nullopt;
}

static SlotSchedule computeSlotSchedule(ArrayRef<GroupDag *> physicalSet, scf::ForOp loop) {
  struct Event {
    GroupDag *group = nullptr;
    Node *node = nullptr;
    Operation *op = nullptr;
    unsigned groupOrder = 0;
  };
  DenseMap<Operation *, unsigned> operationOrder;
  for (auto [index, op] :
       llvm::enumerate(loop.getBody()->without_terminator()))
    operationOrder[&op] = index;
  SlotSchedule result;
  SmallVector<Event, 8> events;
  DenseMap<Node *, unsigned> advancesByAccess;
  for (auto [groupOrder, group] : llvm::enumerate(physicalSet)) {
    forEachNode(*group, [&](Node *node) {
      if (node->kind != Node::Acquire || !node->scheduleAnchor ||
          node->scheduleAnchor->kind != Node::Access || accessEffect(node->scheduleAnchor) != Effect::W)
        return;
      Operation *direct = loop.getBody()->findAncestorOpInBlock(*node->scheduleAnchor->op);
      if (direct == node->scheduleAnchor->op)
        ++advancesByAccess[node->scheduleAnchor];
    });
    forEachNode(*group, [&](Node *node) {
      if (node->kind != Node::Access || !node->op)
        return;
      Operation *direct = loop.getBody()->findAncestorOpInBlock(*node->op);
      if (!direct)
        return;
      if (direct != node->op) {
        result.complete = false;
        return;
      }
      events.push_back(Event{group, node, direct, static_cast<unsigned>(groupOrder)});
    });
  }
  llvm::sort(events, [&](const Event &lhs, const Event &rhs) {
    unsigned lhsOrder = operationOrder.lookup(lhs.op);
    unsigned rhsOrder = operationOrder.lookup(rhs.op);
    return lhsOrder != rhsOrder ? lhsOrder < rhsOrder : lhs.groupOrder < rhs.groupOrder;
  });
  int64_t ordinal = -1;
  int64_t advanceCount = 0;
  DenseMap<GroupDag *, int64_t> lastProducedOrdinal;
  for (const Event &event : events) {
    if (accessEffect(event.node) == Effect::W) {
      unsigned advances = advancesByAccess.lookup(event.node);
      if (advances > 1)
        result.complete = false;
      ordinal += advances;
      advanceCount += advances;
      if (ordinal < 0) {
        result.complete = false;
        continue;
      }
      lastProducedOrdinal[event.group] = ordinal;
      result.ordinalByAccess[event.node] = ordinal;
      continue;
    }
    auto it = lastProducedOrdinal.find(event.group);
    if (it == lastProducedOrdinal.end()) {
      result.complete = false;
      continue;
    }
    result.ordinalByAccess[event.node] = it->second;
  }
  result.advancesPerIteration = advanceCount;
  return result;
}
static int64_t positiveMod(int64_t value, int64_t modulus) {
  int64_t remainder = value % modulus;
  return remainder < 0 ? remainder + modulus : remainder;
}
static std::optional<int64_t>
computeRecurrenceDistance(const SlotSchedule &slots, int64_t depth, Node *src, Node *dst) {
  auto srcIt = slots.ordinalByAccess.find(src);
  auto dstIt = slots.ordinalByAccess.find(dst);
  if (!slots.complete || srcIt == slots.ordinalByAccess.end() ||
      dstIt == slots.ordinalByAccess.end() || slots.advancesPerIteration <= 0)
    return std::nullopt;
  int64_t orbit = depth / std::gcd(depth, slots.advancesPerIteration);
  for (int64_t distance = 1; distance <= orbit; ++distance)
    if (positiveMod(dstIt->second +
                        distance * slots.advancesPerIteration, depth) == positiveMod(srcIt->second, depth))
      return distance;
  return std::nullopt;
}
static bool isExactAliasMultistageGroup(const GroupDag &group) {
  if (!group.isLocal() || group.isCircular() || group.mixedDepthPhysicalAlias ||
      group.pieceTable.members.size() < 2 || group.semaphoreDepth <= 1)
    return false;
  const Member &first = group.pieceTable.members.front();
  return llvm::all_of(group.pieceTable.members, [&](const Member &member) {
    return member.offset == first.offset && member.extent == first.extent && member.type == first.type;
  });
}

static LogicalResult assignAliasedHandoffStageOffsets(GroupDag &group) {
  if (!isExactAliasMultistageGroup(group))
    return success();
  DenseMap<Operation *, SlotSchedule> slotsByLoop;
  bool hasShiftedRelease = false;
  LogicalResult result = success();
  forEachNode(group, [&](Node *release) {
    if (failed(result) || release->kind != Node::Release || !release->sat)
      return;
    Node *src = release->scheduleAnchor;
    Node *dst = release->sat->scheduleAnchor;
    if (!src || !dst || src->kind != Node::Access ||
        dst->kind != Node::Access || src->parent != dst->parent ||
        !src->parent || src->parent->kind != Node::For) {
      result = semaError(src && src->op ? src->op : group.root->op)
               << "staged exact-alias handoff requires direct accesses in one loop body";
      return;
    }
    auto loop = dyn_cast<scf::ForOp>(src->parent->op);
    if (!loop || loop.getBody()->findAncestorOpInBlock(*src->op) != src->op ||
        loop.getBody()->findAncestorOpInBlock(*dst->op) != dst->op) {
      result = semaError(src->op) << "staged exact-alias handoff is not directly represented in "
                  "its ownership loop";
      return;
    }
    auto [it, inserted] = slotsByLoop.try_emplace(loop.getOperation());
    if (inserted)
      it->second = computeSlotSchedule(ArrayRef<GroupDag *>{&group}, loop);
    const SlotSchedule &slots = it->second;
    auto srcIt = slots.ordinalByAccess.find(src);
    auto dstIt = slots.ordinalByAccess.find(dst);
    if (!slots.complete || srcIt == slots.ordinalByAccess.end() ||
        dstIt == slots.ordinalByAccess.end() || slots.advancesPerIteration <= 0) {
      result = semaError(src->op) << "cannot derive staged exact-alias handoff slots";
      return;
    }
    int64_t depth = group.semaphoreDepth;
    int64_t offset = 0;
    if (precedesInChain(release, release->sat)) {
      offset = positiveMod(dstIt->second - srcIt->second, depth);
    } else if (!computeRecurrenceDistance(slots, depth, src, dst)) {
      int64_t nextDst = dstIt->second + slots.advancesPerIteration;
      offset = positiveMod(nextDst - srcIt->second, depth);
    }
    release->stageOffset = offset;
    hasShiftedRelease |= offset != 0;
  });
  if (failed(result))
    return result;
  if (!hasShiftedRelease) {
    forEachNode(group, [&](Node *node) {
      if (node->kind == Node::Release)
        node->stageOffset.reset();
    });
    return success();
  }
  forEachNode(group, [&](Node *node) {
    if (node->kind == Node::Acquire)
      node->stageOffset = 0;
  });
  return success();
}

static LogicalResult addSyncScheduleEdges(MutableArrayRef<GroupDag> groups,
    llvm::MapVector<Operation *, SmallVector<ScheduleEdge, 4>> &edgesByLoop) {
  SmallVector<SmallVector<GroupDag *, 2>, 8> physicalSets;
  DenseMap<GroupDag *, unsigned> setByGroup;
  llvm::MapVector<int64_t, unsigned> circularSetByBuffer;
  for (GroupDag &group : groups) {
    unsigned setIndex;
    if (group.isCircular()) {
      auto [it, inserted] = circularSetByBuffer.insert(
          {group.bufferId, static_cast<unsigned>(physicalSets.size())});
      if (inserted)
        physicalSets.emplace_back();
      setIndex = it->second;
    } else {
      setIndex = physicalSets.size();
      physicalSets.emplace_back();
    }
    physicalSets[setIndex].push_back(&group);
    setByGroup[&group] = setIndex;
  }
  std::map<std::pair<unsigned, Operation *>, SlotSchedule> slotCache;
  for (GroupDag &group : groups) {
    LogicalResult result = success();
    forEachNode(group, [&](Node *release) {
      if (failed(result) || release->kind != Node::Release || !release->sat)
        return;
      Node *acquire = release->sat;
      Operation *src = realScheduleAnchor(release->scheduleAnchor, /*source=*/true);
      Operation *dst = realScheduleAnchor(acquire->scheduleAnchor, /*source=*/false);
      if (!src || !dst)
        return;
      std::optional<LoopAnchorPair> anchors = findCommonScheduledLoop(src, dst);
      if (!anchors || anchors->src == anchors->dst)
        return;
      gpu::StageCluster srcSchedule = gpu::getStageCluster(anchors->src);
      gpu::StageCluster dstSchedule = gpu::getStageCluster(anchors->dst);
      if (!srcSchedule || !dstSchedule)
        return;
      int64_t distance = 0;
      if (!precedesInChain(release, acquire)) {
        unsigned setIndex = setByGroup.lookup(&group);
        auto key = std::make_pair(setIndex, anchors->loop.getOperation());
        auto it = slotCache.find(key);
        if (it == slotCache.end())
          it = slotCache
                   .emplace(key, computeSlotSchedule(physicalSets[setIndex], anchors->loop))
                   .first;
        std::optional<int64_t> recurrenceDistance = computeRecurrenceDistance(
            it->second, group.semaphoreDepth, release->scheduleAnchor, acquire->scheduleAnchor);
        if (!recurrenceDistance) {
          InFlightDiagnostic diag = semaError(anchors->src)
              << "cannot prove physical-slot recurrence distance for a " "scheduled semaphore handoff";
          diag.attachNote(anchors->dst->getLoc()) << "next ownership wave starts here";
          result = failure();
          return;
        }
        distance = *recurrenceDistance;
      }
      int64_t slack = distance + dstSchedule->first - srcSchedule->first;
      if (slack < 0) {
        InFlightDiagnostic diag = semaError(anchors->src) << "fixed loop.stage assignment cannot "
                                     "satisfy semaphore handoff";
        diag << " (source stage " << srcSchedule->first << ", destination stage " << dstSchedule->first
             << ", recurrence distance " << distance << ")";
        diag.attachNote(anchors->dst->getLoc())
            << "destination would execute before the released slot can be " "reacquired";
        result = failure();
        return;
      }
      if (slack == 0)
        edgesByLoop[anchors->loop.getOperation()].push_back(ScheduleEdge{anchors->src, anchors->dst});
    });
    if (failed(result))
      return failure();
  }
  return success();
}
static void addZeroSlackSSAEdges(scf::ForOp loop, SmallVectorImpl<ScheduleEdge> &edges) {
  for (Operation &consumer : loop.getBody()->without_terminator()) {
    gpu::StageCluster consumerSchedule = gpu::getStageCluster(&consumer);
    if (!consumerSchedule)
      continue;
    for (Value operand : getNestedOperands(&consumer)) {
      auto [producer, distance] = triton::getDefiningOpAndDistance(loop, operand);
      if (!producer)
        continue;
      producer = loop.getBody()->findAncestorOpInBlock(*producer);
      if (!producer || producer == &consumer)
        continue;
      gpu::StageCluster producerSchedule = gpu::getStageCluster(producer);
      if (!producerSchedule)
        continue;
      int64_t slack = distance + consumerSchedule->first -
                      producerSchedule->first;
      if (slack == 0)
        edges.push_back(ScheduleEdge{producer, &consumer});
    }
  }
}

static LogicalResult legalizeLoopSchedule(scf::ForOp loop, ArrayRef<ScheduleEdge> edges) {
  SmallVector<Operation *, 32> scheduledOps;
  DenseMap<Operation *, int64_t> cluster;
  for (Operation &op : loop.getBody()->without_terminator()) {
    gpu::StageCluster schedule = gpu::getStageCluster(&op);
    if (!schedule)
      continue;
    scheduledOps.push_back(&op);
    cluster[&op] = schedule->second;
  }
  bool changed = false;
  for (unsigned iteration = 0; iteration <= scheduledOps.size(); ++iteration) {
    changed = false;
    for (const ScheduleEdge &edge : edges) {
      if (!cluster.contains(edge.src) || !cluster.contains(edge.dst))
        continue;
      int64_t separation = edge.src->isBeforeInBlock(edge.dst) ? 0 : 1;
      int64_t required = cluster.lookup(edge.src) + separation;
      if (cluster.lookup(edge.dst) >= required)
        continue;
      cluster[edge.dst] = required;
      changed = true;
    }
    if (!changed)
      break;
    if (iteration == scheduledOps.size())
      return semaError(loop) << "cyclic zero-slack semaphore schedule";
  }
  OpBuilder builder(loop.getContext());
  for (Operation *op : scheduledOps) {
    gpu::StageCluster oldSchedule = gpu::getStageCluster(op);
    int64_t newCluster = cluster.lookup(op);
    if (newCluster == oldSchedule->second)
      continue;
    if (newCluster > std::numeric_limits<int32_t>::max())
      return semaError(op) << "legalized loop.cluster exceeds i32 range";
    gpu::setStageCluster(builder, op, std::make_pair(oldSchedule->first, static_cast<int>(newCluster)));
  }
  return success();
}
static Operation *nextScheduleAnchor(const Node *n) {
  for (const Node *m = n; m; m = m->next)
    if ((m->kind == Node::Access || m->kind == Node::For || m->kind == Node::If) && m->op)
      return m->op;
  return nullptr;
}
static gpu::StageCluster cachedSchedule(const ScheduleCache &cache, const Owner &owner) {
  auto it = cache.find(ownerKey(owner));
  return it == cache.end() ? gpu::StageCluster{} : it->second;
}
static void assignSyncScheduleChain(Node *head, ScheduleCache &cache);
static void assignSyncScheduleRegion(Node *n, ScheduleCache &cache) {
  if (auto forOp = dyn_cast<scf::ForOp>(n->op)) {
    ScheduleCache body = cache;
    assignSyncScheduleChain(n->children[0], body);
    if (!gpu::hasWarpSpecializeTag(forOp))
      cache = std::move(body);
    return;
  }
  ScheduleCache thenCache = cache;
  ScheduleCache elseCache = cache;
  assignSyncScheduleChain(n->children[0], thenCache);
  if (n->children.size() > 1 && n->children[1])
    assignSyncScheduleChain(n->children[1], elseCache);
  cache = std::move(thenCache);
  for (auto &[key, stageCluster] : elseCache)
    cache.try_emplace(key, stageCluster);
}
static void assignSyncScheduleChain(Node *head, ScheduleCache &cache) {
  for (Node *n = head; n; n = n->next) {
    switch (n->kind) {
    case Node::Acquire:
      if (n->owner) {
        if (n->finalPermissionAcquire)
          n->stageCluster = cachedSchedule(cache, n->owner);
        else if (Operation *anchor = nextScheduleAnchor(n->next))
          n->stageCluster = gpu::getStageCluster(anchor);
        else
          n->stageCluster = cachedSchedule(cache, n->owner);
      }
      break;
    case Node::Release:
      if (n->owner)
        n->stageCluster = cachedSchedule(cache, n->owner);
      break;
    case Node::Access:
      if (n->owner) {
        Operation *completion = n->completionAnchor ? n->completionAnchor : n->op;
        cache[ownerKey(n->owner)] = gpu::getStageCluster(completion);
      }
      break;
    case Node::For:
    case Node::If:
      assignSyncScheduleRegion(n, cache);
      break;
    case Node::Enter:
    case Node::Exit:
    case Node::Func:
      break;
    }
  }
}
static void placeFinalAcquireAtLaneExit(Node *n) {
  if (n->kind != Node::Acquire || !n->owner || !n->stageCluster || nextScheduleAnchor(n->next))
    return;
  auto forOp = dyn_cast_or_null<scf::ForOp>(n->parent ? n->parent->op : nullptr);
  if (!forOp || !forOp->hasAttr(triton::kScheduledMaxStageAttrName))
    return;
  auto [stage, cluster] = *n->stageCluster;
  int exitCluster = cluster;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    gpu::StageCluster candidate = gpu::getStageCluster(&op);
    if (!candidate || candidate->first != stage || !gpu::hasPartition(&op))
      continue;
    SetVector<int> partitions = gpu::getPartitionIds(&op);
    if (!partitions.contains(n->owner->first))
      continue;
    exitCluster = std::max(exitCluster, candidate->second);
  }
  n->stageCluster = std::make_pair(stage, exitCluster);
}

LogicalResult finalizeSyncSchedule(MutableArrayRef<GroupDag> groups) {
  llvm::MapVector<Operation *, SmallVector<ScheduleEdge, 4>> edgesByLoop;
  if (failed(assignCircularStageOffsets(groups)))
    return failure();
  for (GroupDag &group : groups)
    if (failed(assignAliasedHandoffStageOffsets(group)))
      return failure();
  if (failed(addMixedDepthAliasScheduleEdges(groups, edgesByLoop)))
    return failure();
  if (failed(addSyncScheduleEdges(groups, edgesByLoop)))
    return failure();
  for (auto &[loopOp, edges] : edgesByLoop) {
    auto loop = cast<scf::ForOp>(loopOp);
    addZeroSlackSSAEdges(loop, edges);
    if (failed(legalizeLoopSchedule(loop, edges)))
      return failure();
  }
  for (GroupDag &g : groups) {
    if (g.root->children.empty())
      continue;
    ScheduleCache cache;
    assignSyncScheduleChain(g.root->children[0], cache);
  }
  for (GroupDag &g : groups)
    forEachNode(g, placeFinalAcquireAtLaneExit);
  return success();
}

LogicalResult buildSyncDag(GroupDag &g, bool useMetaPartitioner, int lowerSemaphoreNumStages,
                           int &numTmemBlocks) {
  SmallVector<EdgeRec> edges;
  if (!g.root->children.empty()) {
    ChainState top; // function chain: games start at bottom (first-touch)
    walkChain(g, g.root->children[0], top, edges, /*underFor=*/false);
  }
  if (failed(buildEdgesAndSemas(g, edges)))
    return failure();
  if (failed(insertEntryAcquires(g)))
    return failure();
  unsigned numComps = numComponents(g);
  if (!g.root->children.empty()) {
    computeCrossings(g, g.root->children[0], numComps);
    pruneDeadIfCrossings(g, g.root->children[0], /*region=*/nullptr);
    if (failed(computeHoldRules(g, g.root->children[0])))
      return failure();
    while (materializeFinalPermissionAcquires(g, g.root->children[0])) {
      refreshCrossingFinals(g, g.root->children[0]);
      if (failed(computeHoldRules(g, g.root->children[0])))
        return failure();
    }
    applyHoldRulePlacement(g, g.root->children[0]); // plan M2: native shape
    computeRequiredParts(g.root->children[0]);
  }
  if (failed(computeBackingPlan(g, useMetaPartitioner, lowerSemaphoreNumStages, numTmemBlocks)))
    return failure();
  if (!g.semas.empty())
    for (Operation *alloc : g.ttDescriptorFedMembers)
      return semaError(alloc) << "managed local_alloc sourced from a tt-form descriptor load — "
                "nvws-insert-allocas must convert this upstream (pipeline " "invariant violated)";
  return verifySyncDag(g);
}
static StringRef asyncOpStr(AsyncOp a) {
  switch (a) {
  case AsyncOp::TC5MMA:
    return "tc5mma";
  case AsyncOp::TMALoad:
    return "tma_load";
  case AsyncOp::CpAsync:
    return "cp_async";
  case AsyncOp::WGMMA:
    return "wgmma";
  case AsyncOp::TMEMCopy:
    return "tmem_copy";
  default:
    return "none";
  }
}
static std::string syncRowLabel(GroupDag &g, const Node *n) {
  std::string s;
  llvm::raw_string_ostream os(s);
  if (n->kind == Node::Acquire)
    os << "a " << getSema(g, n).name;
  else if (n->kind == Node::Release)
    os << "r " << getSema(g, n).name;
  else if (n->kind == Node::For)
    os << "scf.for";
  else if (n->kind == Node::If)
    os << "scf.if";
  return s;
}
static void printThreadInfo(llvm::raw_ostream &os, GroupDag &g, const Node *n) {
  if (!n->requiredParts.empty()) {
    os << " parts{";
    llvm::interleaveComma(n->requiredParts, os);
    os << "}";
  }
  if (!n->crossings.empty()) {
    os << " thread{";
    llvm::interleaveComma(n->crossings, os, [&](const Crossing &c) {
      os << "c" << c.comp << ":" << ownerStr(n->op, c.slotOwner);
    });
    os << "}";
  }
  if (n->kind == Node::For && !n->crossings.empty()) {
    os << " holdrule{";
    llvm::interleaveComma(n->crossings, os, [&](const Crossing &c) {
      os << "c" << c.comp << ":";
      if (c.hold.outcome == Hold::Outcome::CARRIER)
        os << "gated(" << c.hold.reason << ")";
      else if (c.hold.outcome == Hold::Outcome::POINT_OF_USE) {
        os << "pointofuse->";
        if (c.hold.firstToucher && c.hold.firstToucher->op)
          os << c.hold.firstToucher->op->getName().getStringRef();
        else
          os << "?";
      } else
        os << "passthrough-drop";
      if (c.hold.regionTail)
        os << ":regionTail";
      if (c.hold.finalAcquire)
        os << ":finalAcquire";
      if (c.hold.bridgeRelease)
        os << ":entryBridge";
    });
    os << "}";
  }
}
static void printYieldInfo(llvm::raw_ostream &os, GroupDag &g, const Node *exit, const Node *region,
                           unsigned chainIdx) {
  if (!region || region->crossings.empty())
    return;
  os << " yield{";
  llvm::interleaveComma(region->crossings, os, [&](const Crossing &c) {
    os << "c" << c.comp << ": ";
    if (!c.hold.materializesCarrier()) { // native/child-owned: no slot
      os << (c.hold.isChildOwns() ? "drop" : "native");
      return;
    }
    Node *f = chainIdx < c.finals.size() ? c.finals[chainIdx] : nullptr;
    os << (f ? syncRowLabel(g, f) : std::string("pass"));
  });
  os << "}";
}
static void printEffects(llvm::raw_ostream &os, const Node *n) {
  if (n->pieceInfo.empty())
    return;
  os << " effects{";
  llvm::interleaveComma(sortedPieceInfo(n), os, [&](const auto &item) {
    os << "P" << item.first << ":" << (item.second.effect == Effect::W ? "W" : "R");
  });
  os << "}";
}
static void dumpDagChain(GroupDag &g, const Node *head, unsigned depth,
                         const Node *region, unsigned chainIdx, DumpStage stage) {
  auto &os = llvm::errs();
  for (const Node *n = head; n; n = n->next) {
    Operation *anchor = n->parent ? n->parent->op : nullptr;
    switch (n->kind) {
    case Node::Access: {
      if (stage != DumpStage::Sync) {
        for (const Touch &t : n->touches) {
          os << treePrefix(depth) << "|- " << (t.effect == Effect::W ? "W" : "R") << "  m" << t.member
             << "  " << n->op->getName().getStringRef() << " " << ownerStr(n->op, n->owner);
          if (stage == DumpStage::Access && n->completionAnchor != n->op)
            os << " complete=" << n->completionAnchor->getName().getStringRef();
          os << "\n";
        }
        break;
      }
      os << treePrefix(depth) << "|- ";
      llvm::interleaveComma(n->touches, os, [&](const Touch &t) {
        os << (t.effect == Effect::W ? "W" : "R") << " m" << t.member;
      });
      os << "  " << n->op->getName().getStringRef() << " " << ownerStr(n->op, n->owner) << "\n";
      break;
    }
    case Node::Acquire:
    case Node::Release: {
      if (stage != DumpStage::Sync)
        break;
      bool acquire = n->kind == Node::Acquire;
      const Sema &s = getSema(g, n);
      os << treePrefix(depth) << "|- " << (acquire ? "a" : "r") << "  " << s.name;
      if (n->count > 1)
        os << "(" << n->count << ")";
      os << "  " << ownerStr(anchor, n->owner);
      if (acquire && s.isEntry && !n->owner)
        os << "  ; entry";
      if (!acquire) {
        os << " [";
        llvm::interleaveComma(n->payloads, os, [&](AsyncOp p) { os << asyncOpStr(p); });
        os << "]";
      }
      if (n->stageOffset)
        os << "  stage-offset=" << *n->stageOffset;
      os << "\n";
      break;
    }
    case Node::For:
    case Node::If: {
      bool loop = n->kind == Node::For;
      os << treePrefix(depth) << "|- " << (loop ? "scf.for" : "scf.if");
      if (loop && gpu::hasWarpSpecializeTag(n->op))
        os << " (WS, tag=" << *gpu::getWarpSpecializeTag(n->op) << ")";
      if (stage == DumpStage::Access)
        printEffects(os, n);
      else
        printPieceRecord(os, n, loop ? n->op : anchor);
      if (stage == DumpStage::Sync)
        printThreadInfo(os, g, n);
      os << "\n";
      if (loop) {
        dumpDagChain(g, n->children[0], depth + 1, n, 0, stage);
        break;
      }
      os << treePrefix(depth + 1) << "|- then\n";
      dumpDagChain(g, n->children[0], depth + 2, n, 0, stage);
      bool virtualElse = !cast<scf::IfOp>(n->op).elseBlock();
      if (stage != DumpStage::Access || !virtualElse) {
        os << treePrefix(depth + 1) << "|- else" << (virtualElse ? " (virtual)" : "") << "\n";
        if (n->children.size() > 1)
          dumpDagChain(g, n->children[1], depth + 2, n, 1, stage);
      }
      break;
    }
    case Node::Enter:
    case Node::Exit:
      if (stage == DumpStage::Access)
        break;
      os << treePrefix(depth) << (n->kind == Node::Enter ? "|- ENTER" : "|- EXIT");
      printPieceRecord(os, n, anchor);
      if (n->kind == Node::Exit && stage == DumpStage::Sync)
        printYieldInfo(os, g, n, region, chainIdx);
      os << "\n";
      break;
    case Node::Func:
      break;
    }
  }
}

void dumpDagTree(GroupDag &g, DumpStage stage) {
  if (!g.root->children.empty())
    dumpDagChain(g, g.root->children[0], 1, nullptr, 0, stage);
}

void dumpGroupSyncDag(GroupDag &g, triton::FuncOp funcOp) {
  auto &os = llvm::errs();
  os << "SYNC-DAG\n";
  os << "|- func @" << funcOp.getName() << "\n";
  dumpDagTree(g, DumpStage::Sync);
  for (CompId comp = 0; comp < numComponents(g); ++comp) {
    std::string line;
    llvm::raw_string_ostream ls(line);
    bool any = false;
    for (const Sema &s : g.semas) {
      if (s.component != comp)
        continue;
      if (any)
        ls << " ";
      any = true;
      ls << s.name << "{count=" << s.count;
      if (s.isEntry)
        ls << " entry inherit=" << ownerStr(nullptr, s.inheritStamp);
      ls << "}";
    }
    if (any)
      os << "  SEMAS c" << comp << ": " << ls.str() << "\n";
  }
  if (g.semas.empty()) {
    os << "  BACKING: untouched (no semaphores)\n";
    return;
  }
  os << "  BACKING: numStages=" << g.numStages << "\n";
}
} // namespace mlir::triton::nvws_semas
